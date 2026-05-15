/**
 * ecg_preprocess.h
 * ─────────────────
 * ECG signal preprocessing exactly matching the Python training pipeline:
 *
 *  1. Circular ring buffer  (720 samples = 2 × window)
 *  2. DC removal           (IIR high-pass, fc = 0.5 Hz)
 *  3. Bandpass filter      (0.5–40 Hz, 2nd-order biquad)
 *  4. Pan-Tompkins R-peak detector
 *  5. Beat segmentation    (180 samples before + 180 after R-peak = 360 total)
 *  6. Z-score normalisation per beat
 *
 * All coefficients pre-computed for fs = 360 Hz.
 * No dynamic allocation — everything is static.
 */

#pragma once
#include <Arduino.h>
#include <math.h>

// ── Constants ─────────────────────────────────────────────────────────────────
static const int ECG_FS         = 360;    // Hz
static const int ECG_WINDOW_LEN = 360;    // samples per beat
static const int ECG_HALF_WIN   = ECG_WINDOW_LEN / 2;   // 180
static const int ECG_RING_SIZE  = 1440;  // 4 × window

// ── Ring buffer ───────────────────────────────────────────────────────────────
static volatile int   ecg_ring[ECG_RING_SIZE];
static volatile int   ecg_ring_head = 0;
static volatile int   ecg_ring_count = 0;
static volatile int   ecg_raw_last = 0;

// ── Bandpass biquad filter state (0.5–40 Hz @ 360 Hz) ───────────────────────
// Computed via: scipy.signal.butter(2, [0.5, 40], 'bandpass', fs=360, output='sos')
// Verified against Python reference in deployment/verify_filter_coefficients.py
//
// SOS[0]: b=[0.078762, 0.157525, 0.078762]  a=[-1.067092, 0.384234]
// SOS[1]: b=[1.000000, -2.000000, 1.000000]  a=[-1.987664, 0.987742]

struct BiquadState {
  float x1, x2, y1, y2;
};

static const float SOS[2][6] = {
  // b0           b1          b2          a1           a2
  { 0.078762f,  0.157525f,  0.078762f, -1.067092f,  0.384234f },
  { 1.000000f, -2.000000f,  1.000000f, -1.987664f,  0.987742f },
};
static BiquadState bq[2] = {{0,0,0,0},{0,0,0,0}};

// ── Pan-Tompkins derivative + squaring + MWI state ───────────────────────────
static const int PT_MWI_N  = 30;   // ~83 ms window @ 360 Hz
static float pt_deriv_prev = 0.0f;
static float pt_mwi_buf[PT_MWI_N];
static int   pt_mwi_idx    = 0;
static float pt_mwi_sum    = 0.0f;

static const float PT_THRESH_INIT   = 0.20f;
static float pt_signal_level        = PT_THRESH_INIT;
static float pt_noise_level         = PT_THRESH_INIT * 0.5f;
static float pt_threshold           = PT_THRESH_INIT;
static int   pt_refractory          = 0;
static const int PT_REFRACTORY_SAMP = 72;   // 200 ms @ 360 Hz

// ── R-peak ring (timestamps) ──────────────────────────────────────────────────
static const int RPEAK_BUF_SIZE  = 8;
static volatile int rpeak_times[RPEAK_BUF_SIZE];
static volatile int rpeak_head   = 0;
static volatile int rpeak_tail   = 0;
static volatile int rpeak_count  = 0;
static volatile uint32_t sample_clock = 0;   // global sample counter

// Heart rate estimate
static float hr_bpm_est = 0.0f;
static int   rr_buf[4]  = {0,0,0,0};
static int   rr_idx     = 0;

// ── Beat-ready flag ───────────────────────────────────────────────────────────
static volatile bool   beat_ready_flag = false;
static volatile int    beat_rpeak_time = 0;

// ── Internal functions (inline for ISR use) ────────────────────────────────────

inline float biquad_process(int stage, float x) {
  float b0 = SOS[stage][0], b1 = SOS[stage][1], b2 = SOS[stage][2];
  float a1 = SOS[stage][3], a2 = SOS[stage][4];
  float y = b0*x + b1*bq[stage].x1 + b2*bq[stage].x2
              - a1*bq[stage].y1 - a2*bq[stage].y2;
  bq[stage].x2 = bq[stage].x1; bq[stage].x1 = x;
  bq[stage].y2 = bq[stage].y1; bq[stage].y1 = y;
  return y;
}

inline float pan_tompkins(float filtered) {
  // 1. Derivative (5-point)
  float deriv = filtered - pt_deriv_prev;
  pt_deriv_prev = filtered;

  // 2. Squaring
  float sq = deriv * deriv;

  // 3. Moving-window integration
  pt_mwi_sum -= pt_mwi_buf[pt_mwi_idx];
  pt_mwi_buf[pt_mwi_idx] = sq;
  pt_mwi_sum += sq;
  pt_mwi_idx = (pt_mwi_idx + 1) % PT_MWI_N;
  return pt_mwi_sum / PT_MWI_N;
}

// ── Public API ────────────────────────────────────────────────────────────────

void ecg_init() {
  memset((void*)ecg_ring, 0, sizeof(ecg_ring));
  memset(pt_mwi_buf, 0, sizeof(pt_mwi_buf));
  memset((void*)rpeak_times, 0, sizeof(rpeak_times));
  ecg_ring_head = 0; ecg_ring_count = 0; sample_clock = 0;
  rpeak_head = 0; rpeak_tail = 0; rpeak_count = 0;
  beat_ready_flag = false;
  pt_signal_level = PT_THRESH_INIT;
  pt_noise_level  = PT_THRESH_INIT * 0.5f;
  pt_threshold    = PT_THRESH_INIT;
}

void ecg_reset() {
  ecg_init();
  memset((void*)bq, 0, sizeof(bq));
}

// Called from ISR at 360 Hz
void ecg_push_sample(int raw_adc) {
  ecg_raw_last = raw_adc;

  // Store raw in ring buffer
  ecg_ring[ecg_ring_head] = raw_adc;
  ecg_ring_head = (ecg_ring_head + 1) % ECG_RING_SIZE;
  if (ecg_ring_count < ECG_RING_SIZE) ecg_ring_count++;

  // Normalise ADC to [-1, 1] (10-bit: 0–1023)
  float x = (raw_adc - 512.0f) / 512.0f;

  // Bandpass filter (two SOS stages)
  float filtered = biquad_process(0, x);
  filtered        = biquad_process(1, filtered);

  // Pan-Tompkins MWI
  float mwi = pan_tompkins(filtered);

  // Adaptive threshold update
  if (pt_refractory > 0) {
    pt_refractory--;
  } else {
    if (mwi > pt_threshold) {
      // R-peak detected
      pt_signal_level = 0.125f * mwi + 0.875f * pt_signal_level;
      pt_threshold    = pt_noise_level + 0.25f * (pt_signal_level - pt_noise_level);
      pt_refractory   = PT_REFRACTORY_SAMP;

      // Record peak time
      if (rpeak_count < RPEAK_BUF_SIZE) {
        rpeak_times[rpeak_tail] = (int)sample_clock;
        rpeak_tail = (rpeak_tail + 1) % RPEAK_BUF_SIZE;
        rpeak_count++;
      }

      // Compute RR interval for HR
      if (rr_idx > 0) {
        int prev_rr = rr_buf[(rr_idx - 1) % 4];
        int curr_rr = (int)sample_clock - rpeak_times[(rpeak_tail - 2 + RPEAK_BUF_SIZE) % RPEAK_BUF_SIZE];
        rr_buf[rr_idx % 4] = curr_rr;
        rr_idx++;
        // HR = 60 / (mean_RR_in_seconds)
        float mean_rr = 0;
        int n = min(rr_idx, 4);
        for (int i = 0; i < n; i++) mean_rr += rr_buf[i];
        mean_rr /= n;
        if (mean_rr > 0) hr_bpm_est = 60.0f * ECG_FS / mean_rr;
      } else {
        rr_idx = 1;
      }

      // Signal that a beat window is ready IF enough samples buffered
      if (ecg_ring_count >= ECG_WINDOW_LEN && !beat_ready_flag) {
        beat_ready_flag = true;
        beat_rpeak_time = (int)sample_clock;
      }
    } else {
      pt_noise_level  = 0.125f * mwi + 0.875f * pt_noise_level;
      pt_threshold    = pt_noise_level + 0.25f * (pt_signal_level - pt_noise_level);
    }
  }

  sample_clock++;
}

/**
 * ecg_beat_ready()
 * ─────────────────
 * Call from main loop. If a beat is ready, fills `out_window` with
 * 360 z-score-normalised float32 samples and returns true.
 * The window is centred on the R-peak (180 before + 180 after),
 * exactly matching the Python training segmentation.
 */
bool ecg_beat_ready(float out_window[ECG_WINDOW_LEN]) {
  if (!beat_ready_flag) return false;

  // Disable beat flag atomically
  noInterrupts();
  beat_ready_flag = false;
  int peak_t = beat_rpeak_time;
  int ring_h = ecg_ring_head;
  int ring_c = ecg_ring_count;
  interrupts();

  // We need ECG_HALF_WIN samples after the peak already in the buffer.
  // Peak was registered at peak_t; current clock is sample_clock.
  int samples_since_peak = (int)(sample_clock - peak_t);
  if (samples_since_peak < ECG_HALF_WIN) {
    // Not enough post-peak samples yet — defer
    beat_ready_flag = true;
    return false;
  }

  // Extract 360 raw ADC samples centred on R-peak
  // ring_h points to the NEXT write slot.
  // The sample at (ring_h - samples_since_peak + ECG_HALF_WIN) is the peak.
  // Walk back from current head:
  //   window_start_offset = samples_since_peak + ECG_HALF_WIN - 1
  //   window_end_offset   = samples_since_peak - ECG_HALF_WIN - 1

  float raw_f[ECG_WINDOW_LEN];
  float mean = 0.0f, var = 0.0f;

  int start_ago = samples_since_peak + ECG_HALF_WIN;  // samples ago for window[0]
  for (int i = 0; i < ECG_WINDOW_LEN; i++) {
    int ago = start_ago - i;
    int idx = ((int)ring_h - ago + ECG_RING_SIZE * 4) % ECG_RING_SIZE;
    raw_f[i] = (float)ecg_ring[idx];
    mean += raw_f[i];
  }
  mean /= ECG_WINDOW_LEN;

  for (int i = 0; i < ECG_WINDOW_LEN; i++) {
    float d = raw_f[i] - mean;
    var += d * d;
  }
  float std_dev = sqrtf(var / ECG_WINDOW_LEN + 1e-8f);

  // Z-score normalise — EXACTLY matches Python: (x - mean) / std
  for (int i = 0; i < ECG_WINDOW_LEN; i++) {
    out_window[i] = (raw_f[i] - mean) / std_dev;
  }

  return true;
}

int   ecg_last_raw()     { return ecg_raw_last; }
float ecg_heart_rate()   { return hr_bpm_est; }
