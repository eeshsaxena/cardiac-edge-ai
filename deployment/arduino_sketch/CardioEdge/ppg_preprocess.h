/**
 * ppg_preprocess.h
 * ─────────────────
 * MAX30102 PPG sensor driver + signal preprocessing.
 *
 * MAX30102 is an I2C pulse-oximeter / heart-rate sensor.
 *   - Reads IR channel @ 100 Hz (native)
 *   - DC removal via IIR high-pass (fc = 0.5 Hz)
 *   - Low-pass filter at 4 Hz (anti-alias for heartbeat envelope)
 *   - Circular buffer of PPG_WINDOW=360 samples (for fusion model)
 *
 * If the sensor is absent the ppg_init() returns false and PPG
 * data is simply zeroed — ECG-only inference still works.
 *
 * Register map references: MAX30102 datasheet Rev 1, pp. 18-30.
 */

#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// ── I2C address & key registers ──────────────────────────────────────────────
static const uint8_t MAX30102_ADDR           = 0x57;
static const uint8_t REG_INT_STATUS1         = 0x00;
static const uint8_t REG_FIFO_WR_PTR        = 0x04;
static const uint8_t REG_FIFO_RD_PTR        = 0x06;
static const uint8_t REG_FIFO_DATA          = 0x07;
static const uint8_t REG_FIFO_CONFIG        = 0x08;
static const uint8_t REG_MODE_CONFIG        = 0x09;
static const uint8_t REG_SPO2_CONFIG        = 0x0A;
static const uint8_t REG_LED1_PA            = 0x0C;  // Red LED
static const uint8_t REG_LED2_PA            = 0x0D;  // IR LED
static const uint8_t REG_PART_ID            = 0xFF;

// ── PPG buffer (matches ECG window length for fusion model input) ─────────────
static const int PPG_NATIVE_FS   = 100;   // Hz — MAX30102 sample rate
static const int PPG_WINDOW_LEN  = 360;   // final buffer after upsampling to 360

static float ppg_raw_buf[512];            // ring buffer at 100 Hz
static int   ppg_raw_head   = 0;
static int   ppg_raw_count  = 0;

// Filtered buffer for output
static float ppg_out_buf[PPG_WINDOW_LEN];
static bool  ppg_available  = false;
static bool  ppg_sensor_ok  = false;

// ── IIR filter state (DC removal + anti-alias LP) ────────────────────────────
// HP: fc=0.5 Hz @ 100 Hz,  y[n] = 0.9690*y[n-1] + x[n] - x[n-1]
static const float HP_COEFF = 0.9690f;
static float ppg_hp_prev_x  = 0.0f;
static float ppg_hp_prev_y  = 0.0f;

// LP: fc=4 Hz @ 100 Hz,  1st-order RC
static const float LP_ALPHA = 0.2199f;   // 2*pi*4/100 / (1 + 2*pi*4/100)
static float ppg_lp_prev    = 0.0f;

// ── I2C helpers ───────────────────────────────────────────────────────────────
static void max30102_write(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(reg); Wire.write(val);
  Wire.endTransmission();
}

static uint8_t max30102_read8(uint8_t reg) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MAX30102_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

static bool max30102_read_fifo(uint32_t* ir_out) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(REG_FIFO_DATA);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MAX30102_ADDR, (uint8_t)6);
  if (Wire.available() < 6) return false;

  uint32_t red_raw = ((uint32_t)Wire.read() << 16) |
                     ((uint32_t)Wire.read() <<  8) |
                      (uint32_t)Wire.read();
  uint32_t ir_raw  = ((uint32_t)Wire.read() << 16) |
                     ((uint32_t)Wire.read() <<  8) |
                      (uint32_t)Wire.read();
  red_raw &= 0x3FFFF;  // 18-bit
  ir_raw  &= 0x3FFFF;
  *ir_out = ir_raw;
  return true;
}

// ── Public API ────────────────────────────────────────────────────────────────

bool ppg_init() {
  Wire.begin();
  delay(10);

  // Check Part ID (MAX30102 = 0x15)
  uint8_t pid = max30102_read8(REG_PART_ID);
  if (pid != 0x15) {
    ppg_sensor_ok = false;
    return false;
  }

  // Reset
  max30102_write(REG_MODE_CONFIG, 0x40);
  delay(100);

  // FIFO: sample avg = 4, FIFO rollover on, FIFO almost full = 17
  max30102_write(REG_FIFO_CONFIG, 0b00100000 | 0b00010000 | 15);

  // Mode: SpO2 mode (Red + IR)
  max30102_write(REG_MODE_CONFIG, 0x03);

  // SpO2: ADC range = 4096 nA, sample rate = 100 Hz, pulse width = 411 µs
  max30102_write(REG_SPO2_CONFIG, 0b00100111);

  // LED pulse amplitude: ~10 mA
  max30102_write(REG_LED1_PA, 0x24);
  max30102_write(REG_LED2_PA, 0x24);

  // Clear FIFO
  max30102_write(REG_FIFO_WR_PTR, 0);
  max30102_write(REG_FIFO_RD_PTR, 0);

  ppg_sensor_ok = true;
  memset(ppg_raw_buf, 0, sizeof(ppg_raw_buf));
  memset(ppg_out_buf, 0, sizeof(ppg_out_buf));
  return true;
}

/**
 * ppg_update()
 * ─────────────
 * Call from main loop as often as possible.
 * Drains the MAX30102 FIFO, applies DC removal + LP filter,
 * stores filtered samples in the ring buffer.
 */
void ppg_update() {
  if (!ppg_sensor_ok) return;

  // How many samples in FIFO?
  uint8_t wr_ptr = max30102_read8(REG_FIFO_WR_PTR);
  uint8_t rd_ptr = max30102_read8(REG_FIFO_RD_PTR);
  int avail = ((int)wr_ptr - (int)rd_ptr + 32) % 32;

  for (int s = 0; s < avail; s++) {
    uint32_t ir;
    if (!max30102_read_fifo(&ir)) break;

    // Normalise to [0, 1]
    float x = (float)ir / 262143.0f;  // 18-bit max

    // DC removal (high-pass)
    float hp = HP_COEFF * ppg_hp_prev_y + x - ppg_hp_prev_x;
    ppg_hp_prev_x = x;
    ppg_hp_prev_y = hp;

    // Anti-alias low-pass
    float lp = LP_ALPHA * hp + (1.0f - LP_ALPHA) * ppg_lp_prev;
    ppg_lp_prev = lp;

    // Store in ring buffer
    ppg_raw_buf[ppg_raw_head] = lp;
    ppg_raw_head = (ppg_raw_head + 1) % 512;
    if (ppg_raw_count < 512) ppg_raw_count++;
  }
}

/**
 * ppg_get_window()
 * ─────────────────
 * Returns PPG window upsampled to PPG_WINDOW_LEN (360) samples
 * via linear interpolation, z-score normalised — ready for fusion model.
 * Returns false if not enough data yet.
 */
bool ppg_get_window(float out[PPG_WINDOW_LEN]) {
  if (!ppg_sensor_ok || ppg_raw_count < PPG_NATIVE_FS) {
    memset(out, 0, PPG_WINDOW_LEN * sizeof(float));
    return false;
  }

  // Grab last PPG_NATIVE_FS samples (1 second at 100 Hz)
  float src[PPG_NATIVE_FS];
  for (int i = 0; i < PPG_NATIVE_FS; i++) {
    int idx = ((int)ppg_raw_head - PPG_NATIVE_FS + i + 512) % 512;
    src[i] = ppg_raw_buf[idx];
  }

  // Linear upsampling 100 → 360
  float ratio = (float)(PPG_NATIVE_FS - 1) / (float)(PPG_WINDOW_LEN - 1);
  float mean = 0.0f, var = 0.0f;
  float tmp[PPG_WINDOW_LEN];
  for (int i = 0; i < PPG_WINDOW_LEN; i++) {
    float pos  = i * ratio;
    int   lo   = (int)pos;
    float frac = pos - lo;
    int   hi   = min(lo + 1, PPG_NATIVE_FS - 1);
    tmp[i]     = src[lo] * (1.0f - frac) + src[hi] * frac;
    mean += tmp[i];
  }
  mean /= PPG_WINDOW_LEN;
  for (int i = 0; i < PPG_WINDOW_LEN; i++) {
    float d = tmp[i] - mean; var += d * d;
  }
  float std_dev = sqrtf(var / PPG_WINDOW_LEN + 1e-8f);
  for (int i = 0; i < PPG_WINDOW_LEN; i++) {
    out[i] = (tmp[i] - mean) / std_dev;
  }
  return true;
}
