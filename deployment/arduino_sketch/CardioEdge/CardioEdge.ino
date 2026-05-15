/**
 * CardioEdge — 5-class Arrhythmia Detector
 * Arduino Nano 33 BLE  (nRF52840, 64 MHz Cortex-M4F)
 *
 * Hardware:
 *   AD8232  ECG module  → A0  (analog), LO+ → D10, LO- → D11
 *   MAX30102 PPG module → I2C (SDA=A4, SCL=A5)
 *   Status LED          → D13 (built-in)
 *   Alert LED (red)     → D2
 *
 * Signal pipeline (mirrors Python training exactly):
 *   ECG: 360 Hz ADC → bandpass 0.5–40 Hz → R-peak detect →
 *        360-sample beat window → z-score norm → TFLite INT8
 *
 * Required Arduino Libraries (install via Library Manager):
 *   - Arduino_TensorFlowLite  ≥ 2.4.0
 *   - SparkFun MAX3010x Sensor Library  ≥ 1.1.2
 *   - ArduinoFFT (optional, for frequency display)
 *
 * Classes: 0=N  1=AF  2=VT  3=PVC  4=LBBB
 */

#include <Arduino.h>
#include <Wire.h>

// TFLite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Our generated model & helpers  (all files must be in the same sketch folder)
#include "student_kd_model.h"   // 23.4 KB INT8 TFLite C array
#include "ecg_preprocess.h"     // ECG sampling + bandpass + Pan-Tompkins
#include "ppg_preprocess.h"     // MAX30102 driver + PPG normalisation
#include "signal_buffer.h"      // ISR-safe circular buffers

// ── Pins ──────────────────────────────────────────────────────────────────────
static const int PIN_ECG_IN   = A0;
static const int PIN_LO_PLUS  = 10;   // AD8232 lead-off detection
static const int PIN_LO_MINUS = 11;
static const int PIN_LED_OK   = LED_BUILTIN;  // D13 green
static const int PIN_LED_WARN = 2;            // D2  red

// ── TFLite globals ────────────────────────────────────────────────────────────
namespace {
  const int kTensorArenaSize = 24 * 1024;   // 24 KB  (model needs ~18 KB)
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];

  const tflite::Model*        model_ptr   = nullptr;
  tflite::AllOpsResolver      resolver;
  tflite::MicroInterpreter*   interpreter = nullptr;
  TfLiteTensor*               input_tensor  = nullptr;
  TfLiteTensor*               output_tensor = nullptr;
}

// ── Class labels ─────────────────────────────────────────────────────────────
static const char* CLASS_NAMES[] = {"N", "AF", "VT", "PVC", "LBBB"};
static const char* CLASS_DESC[]  = {
  "Normal",
  "Atrial Fibrillation",
  "Ventricular Tachycardia",
  "Premature Ventricular Contraction",
  "Left Bundle Branch Block"
};

// ── State ────────────────────────────────────────────────────────────────────
static bool     leads_ok        = false;
static bool     debug_stream    = false;    // toggle raw ADC via 'd' cmd
static float    beat_window[ECG_WINDOW_LEN];
static uint32_t beat_count      = 0;
static uint32_t alert_count     = 0;

// ── Forward declarations ──────────────────────────────────────────────────────
bool  run_inference(const float* ecg_samples);
void  print_result(int cls, const float* probs);
void  handle_serial_commands();       // 'r' reset / 'd' debug / 's' stats
void  setup_timer();                  // 360 Hz ECG sampling via hardware timer

// ── ISR: 360 Hz ECG sample ───────────────────────────────────────────────────
// nRF52840 uses NRF_TIMER1 for high-res periodic ISR
extern "C" void TIMER1_IRQHandler(void) {
  if (NRF_TIMER1->EVENTS_COMPARE[0]) {
    NRF_TIMER1->EVENTS_COMPARE[0] = 0;
    int raw = analogRead(PIN_ECG_IN);  // 0..1023  (10-bit ADC)
    ecg_push_sample(raw);              // → circular buffer → filter → R-peak detect
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}  // wait up to 3 s for Serial monitor

  Serial.println(F("\n========================================"));
  Serial.println(F("  CardioEdge v1.0 — Arrhythmia Detector"));
  Serial.println(F("  Arduino Nano 33 BLE + TFLite INT8"));
  Serial.println(F("========================================\n"));

  // ── GPIO ──────────────────────────────────────────────────────────────────
  pinMode(PIN_LO_PLUS,  INPUT);
  pinMode(PIN_LO_MINUS, INPUT);
  pinMode(PIN_LED_OK,   OUTPUT);
  pinMode(PIN_LED_WARN, OUTPUT);
  digitalWrite(PIN_LED_OK,   LOW);
  digitalWrite(PIN_LED_WARN, LOW);

  // ── MAX30102 PPG ──────────────────────────────────────────────────────────
  Serial.print(F("Initialising MAX30102 PPG sensor ... "));
  if (!ppg_init()) {
    Serial.println(F("FAILED — check I2C wiring"));
    // PPG is optional — ECG-only inference still works
  } else {
    Serial.println(F("OK"));
  }

  // ── Load TFLite model ─────────────────────────────────────────────────────
  Serial.print(F("Loading TFLite model ... "));
  model_ptr = tflite::GetModel(student_kd_model);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("SCHEMA MISMATCH — rebuild model"));
    while (true) {}
  }

  static tflite::MicroInterpreter static_interpreter(
      model_ptr, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println(F("AllocateTensors FAILED"));
    while (true) {}
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Validate shapes: input  (1, 360, 1)
  Serial.print(F("OK — input shape: ["));
  for (int i = 0; i < input_tensor->dims->size; i++) {
    Serial.print(input_tensor->dims->data[i]);
    if (i < input_tensor->dims->size - 1) Serial.print(F(", "));
  }
  Serial.println(F("]"));
  Serial.print(F("Arena used: "));
  Serial.print(interpreter->arena_used_bytes() / 1024.0f, 1);
  Serial.println(F(" KB / 24 KB"));

  // ── ECG preprocessing engine ──────────────────────────────────────────────
  ecg_init();

  // ── 360 Hz hardware timer ─────────────────────────────────────────────────
  setup_timer();

  Serial.println(F("\nSystem ready. Attach electrodes and breathe normally.\n"));
  Serial.println(F("Format: [beat#] CLASS (description)  conf=XX%  HR=XXbpm\n"));

  digitalWrite(PIN_LED_OK, HIGH);
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  // ── Check lead-off ────────────────────────────────────────────────────────
  bool lo_plus  = digitalRead(PIN_LO_PLUS);
  bool lo_minus = digitalRead(PIN_LO_MINUS);
  leads_ok = !(lo_plus || lo_minus);

  if (!leads_ok) {
    digitalWrite(PIN_LED_WARN, HIGH);
    if (millis() % 1000 < 50) {
      Serial.println(F("[WARN] Lead-off detected — reattach electrodes"));
    }
    ecg_reset();   // flush stale buffers
    delay(50);
    return;
  }
  digitalWrite(PIN_LED_WARN, LOW);

  // ── Check if ECG engine has a complete beat ready ─────────────────────────
  if (ecg_beat_ready(beat_window)) {
    beat_count++;

    // ── PPG: read current SpO2/PPG value (non-blocking) ───────────────────
    // Not used for the student_kd model (ECG-only INT8), kept for
    // future fusion model deployment.
    ppg_update();

    // ── Run TFLite inference ───────────────────────────────────────────────
    float probs[5];
    if (run_inference(beat_window)) {
      int pred = 0;
      for (int i = 1; i < 5; i++) {
        if (probs[i] > probs[pred]) pred = i;
      }
      print_result(pred, probs);

      // Alert LED for dangerous rhythms (VT, PVC, LBBB)
      if (pred == 2 || pred == 3 || pred == 4) {
        alert_count++;
        for (int f = 0; f < 3; f++) {
          digitalWrite(PIN_LED_WARN, HIGH); delay(80);
          digitalWrite(PIN_LED_WARN, LOW);  delay(80);
        }
      }
    }
  }

  // ── Stream raw ADC for Serial Plotter (toggled via 'd' command) ────────────
  if (debug_stream) {
    Serial.println(ecg_last_raw());
  }

  // ── Handle Serial commands ────────────────────────────────────────────────
  handle_serial_commands();

  delay(1);
}

// ─────────────────────────────────────────────────────────────────────────────
void handle_serial_commands() {
  if (!Serial.available()) return;
  char cmd = Serial.read();
  switch (cmd) {
    case 'r':
      ecg_reset();
      beat_count = 0; alert_count = 0;
      Serial.println(F("[CMD] ECG engine reset."));
      break;
    case 'd':
      debug_stream = !debug_stream;
      Serial.print(F("[CMD] Raw ADC stream: "));
      Serial.println(debug_stream ? F("ON") : F("OFF"));
      break;
    case 's':
      Serial.print(F("[STATS] Beats: "));
      Serial.print(beat_count);
      Serial.print(F("  Alerts: "));
      Serial.print(alert_count);
      Serial.print(F("  HR: "));
      Serial.print(ecg_heart_rate(), 0);
      Serial.print(F(" bpm  Arena: "));
      Serial.print(interpreter->arena_used_bytes() / 1024.0f, 1);
      Serial.println(F(" KB"));
      break;
  }
  // flush rest of line
  while (Serial.available()) Serial.read();
}

// ─────────────────────────────────────────────────────────────────────────────
bool run_inference(const float* ecg_samples) {
  // Copy 360 float32 samples into input tensor  shape (1, 360, 1)
  float* inp = input_tensor->data.f;
  for (int i = 0; i < ECG_WINDOW_LEN; i++) {
    inp[i] = ecg_samples[i];
  }

  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println(F("[ERR] Inference failed"));
    return false;
  }

  // Copy outputs — model returns float32 softmax
  float* out = output_tensor->data.f;
  for (int i = 0; i < 5; i++) {
    // probs stored in local but print_result reads from output tensor directly
    (void)out[i];
  }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
void print_result(int cls, const float* probs) {
  float* out    = output_tensor->data.f;
  float  conf   = out[cls] * 100.0f;
  float  hr_bpm = ecg_heart_rate();

  Serial.print(F("["));
  Serial.print(beat_count);
  Serial.print(F("] "));
  Serial.print(CLASS_NAMES[cls]);
  Serial.print(F(" — "));
  Serial.print(CLASS_DESC[cls]);
  Serial.print(F("  conf="));
  Serial.print(conf, 1);
  Serial.print(F("%  HR="));
  Serial.print(hr_bpm, 0);
  Serial.print(F("bpm  probs=["));
  for (int i = 0; i < 5; i++) {
    Serial.print(out[i], 3);
    if (i < 4) Serial.print(F(" "));
  }
  Serial.println(F("]"));
}

// ─────────────────────────────────────────────────────────────────────────────
// 360 Hz hardware timer using nRF52840 TIMER1
// Period = 1 / 360 Hz ≈ 2777 µs.  TIMER1 at 1 MHz → 2777 ticks.
void setup_timer() {
  NRF_TIMER1->TASKS_STOP   = 1;
  NRF_TIMER1->TASKS_CLEAR  = 1;
  NRF_TIMER1->MODE         = TIMER_MODE_MODE_Timer;
  NRF_TIMER1->BITMODE      = TIMER_BITMODE_BITMODE_32Bit;
  NRF_TIMER1->PRESCALER    = 4;             // 16 MHz / 2^4 = 1 MHz → 1 µs tick
  NRF_TIMER1->CC[0]        = 2778;          // 1,000,000 / 360 ≈ 2778 µs
  NRF_TIMER1->SHORTS       = TIMER_SHORTS_COMPARE0_CLEAR_Msk;
  NRF_TIMER1->INTENSET     = TIMER_INTENSET_COMPARE0_Msk;
  NVIC_SetPriority(TIMER1_IRQn, 3);
  NVIC_EnableIRQ(TIMER1_IRQn);
  NRF_TIMER1->TASKS_START  = 1;
  Serial.println(F("360 Hz ECG sampling timer started."));
}
