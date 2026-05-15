/**
 * filter_verify.ino
 * ─────────────────
 * Plays back 5 real MIT-BIH beats (N, AF, VT, PVC, LBBB) through the
 * EXACT same Arduino preprocessing pipeline as CardioEdge.ino.
 *
 * Step-by-step verification:
 *  1. Upload this sketch to Arduino Nano 33 BLE
 *  2. Run: python deployment/verify_filter_coefficients.py --port COM3
 *  3. Confirm per-class predictions match Python TFLite ground truth:
 *       Beat 0 (N):    pred=N     ~67% conf
 *       Beat 2 (VT):   pred=VT   ~89% conf
 *       Beat 4 (LBBB): pred=LBBB ~85% conf
 *  4. When R-peak positions are within ±2 samples of Python, switch to CardioEdge.ino
 */

#include "ecg_preprocess.h"
#include "mit_playback.h"   // auto-generated: 1800 samples (5 × 360 beat)

static const char* CLASS_NAMES_FV[] = {"N", "AF", "VT", "PVC", "LBBB"};
static const int   BEAT_SIZE        = 360;
static const int   N_BEATS          = 5;

static int  sample_idx  = 0;
static bool debug_raw   = true;   // stream raw ADC to serial plotter
static float beat_buf[ECG_WINDOW_LEN];
static int  beat_idx    = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  Serial.println(F("# filter_verify — MIT-BIH 5-beat playback"));
  Serial.println(F("# Connect to: python deployment/verify_filter_coefficients.py"));
  Serial.println(F("# Beat boundaries: N=0-359, AF=360-719, VT=720-1079, PVC=1080-1439, LBBB=1440-1799"));
  ecg_init();
}

void loop() {
  if (sample_idx >= N_PLAYBACK) {
    Serial.println(F("# === Playback complete. Restarting in 2s ==="));
    delay(2000);
    sample_idx = 0;
    beat_idx   = 0;
    ecg_init();
    return;
  }

  int raw = (int)pgm_read_word(&MIT_PLAYBACK[sample_idx]);
  ecg_push_sample(raw);

  // Announce which beat we're entering
  if (sample_idx % BEAT_SIZE == 0) {
    int b = sample_idx / BEAT_SIZE;
    if (b < N_BEATS) {
      Serial.print(F("# --- Beat ")); Serial.print(b);
      Serial.print(F(" (true class: "));
      Serial.print(CLASS_NAMES_FV[b]);
      Serial.println(F(") ---"));
    }
  }

  // Stream raw ADC for Python plotter
  if (debug_raw) {
    Serial.println(raw);
  }

  // Check if a beat window is ready (R-peak detected)
  if (ecg_beat_ready(beat_buf)) {
    // Print z-score normalised window as CSV for Python to classify
    Serial.print(F("#BEAT_WIN:"));
    for (int i = 0; i < ECG_WINDOW_LEN; i++) {
      Serial.print(beat_buf[i], 4);
      if (i < ECG_WINDOW_LEN - 1) Serial.print(',');
    }
    Serial.println();
    Serial.print(F("# HR estimate: "));
    Serial.print(ecg_heart_rate(), 1);
    Serial.println(F(" bpm"));
    beat_idx++;
  }

  sample_idx++;
  delayMicroseconds(2778);  // 1 / 360 Hz
}
