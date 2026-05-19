// ======================================================
// XIAO ESP32-S3 SENSE
// FULL AUDIO ML — WAKE WORD CLASSIFICATION v13
//
//
//
// SD card layout:
//   /audio/<ClassName>/clip_NNNNN.wav   — training clips
//   /header/myWeights.bin               — saved weights (binary)
//   /header/myWeights.h                 — saved weights (C header)
//
// INPUT:  1-second PDM @ 16 kHz -> 40-band Mel spectrogram (40x32 = 1280)
// MODEL:  Conv1(3x3,4) -> MaxPool -> Conv2(3x3,8) -> Dense -> Softmax
//
// Arduino IDE Tools:
//   Board:           XIAO_ESP32S3
//   USB CDC On Boot: Enabled
//   PSRAM:           OPI PSRAM
//
// By Jeremy Ellis  https://github.com/hpssjellis
// MIT License
// ======================================================

//#define USE_BAKED_WEIGHTS
#ifdef USE_BAKED_WEIGHTS
  #include "myWeights.h"
#endif

#include "ESP_I2S.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <math.h>
#include <U8g2lib.h>
#include <Wire.h>
#include "freertos/FreeRTOS.h"   // for vTaskDelay(0) — yields to scheduler/idle task

U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// CLASS LABELS — edit to match your wake words
// ======================================================
#define NUM_CLASSES 3
String myClassLabels[NUM_CLASSES] = {"0Silence", "1Yes", "2No"};
const int myTotalItems = NUM_CLASSES + 2;   // classes + Train + Infer

// ======================================================
// HYPERPARAMETERS
// ======================================================
float LEARNING_RATE    = 0.003f;   //= 0.0003f;
int   BATCH_SIZE       = 12;          // 6;
int   TARGET_EPOCHS    = 20;         //  20;
int   VALIDATION_CLIPS = 4l;      // Note this is 4 and "l" for long data type (a 64-bit integer) 
                                  // instead of a standard int (a 32-bit integer).
                                  // clips per class held out for validation
// ======================================================
// AUDIO / SPECTROGRAM CONSTANTS
// ======================================================
#define SAMPLE_RATE   16000
#define CLIP_SECONDS  1
#define CLIP_SAMPLES  (SAMPLE_RATE * CLIP_SECONDS)   // 16000 samples
#define MEL_BINS      40
#define MEL_FRAMES    32
#define FRAME_SIZE    (CLIP_SAMPLES / MEL_FRAMES)    // 500 samples/frame
#define FFT_SIZE      512

#define INPUT_W       MEL_BINS    // 40  (spectrogram "width")
#define INPUT_H       MEL_FRAMES  // 32  (spectrogram "height")
#define INPUT_CH      1           // single-channel (greyscale spectrogram)

// ======================================================
// CNN ARCHITECTURE  (mirrors vision v44, 1-channel input)
// ======================================================
#define CONV1_FILTERS  4
#define CONV1_WEIGHTS  (3 * 3 * INPUT_CH * CONV1_FILTERS)        //  36

#define CONV2_FILTERS  8
#define CONV2_WEIGHTS  (3 * 3 * CONV1_FILTERS * CONV2_FILTERS)   // 288

#define CONV1_OUT_H   (INPUT_H - 2)         // 30
#define CONV1_OUT_W   (INPUT_W - 2)         // 38
#define POOL1_OUT_H   (CONV1_OUT_H / 2)     // 15
#define POOL1_OUT_W   (CONV1_OUT_W / 2)     // 19
#define CONV2_OUT_H   (POOL1_OUT_H - 2)     // 13
#define CONV2_OUT_W   (POOL1_OUT_W - 2)     // 17

#define FLATTENED_SIZE  (CONV2_OUT_H * CONV2_OUT_W * CONV2_FILTERS)  // 1768
#define OUTPUT_WEIGHTS  (FLATTENED_SIZE * NUM_CLASSES)

// ======================================================
// TOUCH INPUT
// ======================================================
const int myThresholdPress   = 1100;
const int myThresholdRelease = 900;

struct TouchState {
  bool          isTouching      = false;
  int           tapCount        = 0;
  unsigned long firstTapTime    = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime   = 0;
  const unsigned long tapWindow     = 800;
  const int           longPressTaps = 3;
  const unsigned long debounceDelay = 50;
};
TouchState myTouch;

// ======================================================
// SYSTEM STATE
// ======================================================
unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime      = 0;
const int     myTapCooldown      = 250;
int  myMenuIndex      = 1;
bool myIsSelected     = false;
bool myWeightsTrained = false;
bool mySDavailable    = false;

// ======================================================
// AUDIO
// ======================================================
I2SClass myI2S;
int16_t* myAudioBuffer = nullptr;

// ======================================================
// SPECTROGRAM SCRATCH BUFFERS — allocated once in PSRAM
// ======================================================
float* mySpectroFrame = nullptr;   // FFT_SIZE floats
float* mySpectroPower = nullptr;   // FFT_SIZE/2+1 floats

// ======================================================
// ML WEIGHT / ACTIVATION / GRADIENT BUFFERS (PSRAM)
// ======================================================
float* myInputBuffer  = nullptr;

float* myConv1_w  = nullptr;  float* myConv1_b  = nullptr;
float* myConv2_w  = nullptr;  float* myConv2_b  = nullptr;
float* myOutput_w = nullptr;  float* myOutput_b = nullptr;

float* myConv1_w_grad  = nullptr; float* myConv1_b_grad  = nullptr;
float* myConv2_w_grad  = nullptr; float* myConv2_b_grad  = nullptr;
float* myOutput_w_grad = nullptr; float* myOutput_b_grad = nullptr;

float* myConv1_w_m = nullptr; float* myConv1_w_v = nullptr;
float* myConv1_b_m = nullptr; float* myConv1_b_v = nullptr;
float* myConv2_w_m = nullptr; float* myConv2_w_v = nullptr;
float* myConv2_b_m = nullptr; float* myConv2_b_v = nullptr;
float* myOutput_w_m = nullptr; float* myOutput_w_v = nullptr;
float* myOutput_b_m = nullptr; float* myOutput_b_v = nullptr;

float* myConv1_output = nullptr;
float* myPool1_output = nullptr;
float* myConv2_output = nullptr;
float* myDense_output = nullptr;

// Pool1 argmax: stores which of the 4 cells (0-3) in each 2x2 window
// was the maximum, so the backward pass can route gradients correctly
// without fragile float equality comparisons.
// Size = POOL1_OUT_H * POOL1_OUT_W * CONV1_FILTERS
uint8_t* myPool1_argmax = nullptr;

float* myDense_grad = nullptr;
float* myConv2_grad = nullptr;
float* myPool1_grad = nullptr;
float* myConv1_grad = nullptr;

struct TrainingItem { String path; int label; };
std::vector<TrainingItem> myTrainingData;

// ======================================================
// UTILITY
// ======================================================
inline float clip_value(float v, float mn = -100, float mx = 100) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}
inline float leaky_relu(float x)       { return x > 0 ? x : 0.1f * x; }
inline float leaky_relu_deriv(float x) { return x > 0 ? 1.0f : 0.1f; }

// ======================================================
// TOUCH INPUT
// ======================================================
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) { sum += analogRead(A0); delayMicroseconds(100); }
  return sum / 3;
}

void myResetTouchState() {
  myTouch.isTouching      = false;
  myTouch.tapCount        = 0;
  myTouch.firstTapTime    = 0;
  myTouch.lastReleaseTime = 0;
  myTouch.lastCheckTime   = 0;
}

void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;
  int val = myReadTouch();
  bool active = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);
  if (active && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
    } else {
      myTouch.tapCount     = 1;
      myTouch.firstTapTime = now;
    }
  }
  if (!active && myTouch.isTouching) {
    myTouch.isTouching      = false;
    myTouch.lastReleaseTime = now;
  }
}

int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching &&
      now - myTouch.firstTapTime > myTouch.tapWindow) {
    int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
    myResetTouchState();
    return result;
  }
  return 0;
}

void myCheckTouchBackground() { myUpdateTouchState(); }

int myPeekTouchAction() {
  myUpdateTouchState();
  if (myTouch.tapCount > 0 && !myTouch.isTouching &&
      millis() - myTouch.firstTapTime > myTouch.tapWindow)
    return (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
  return 0;
}

// ======================================================
// RECORD AND SAVE A CLIP TO SD
// ======================================================
bool mySaveClipToSD(const String& path) {
  size_t   wav_size   = 0;
  uint8_t* wav_buffer = myI2S.recordWAV(CLIP_SECONDS, &wav_size);
  if (!wav_buffer || wav_size == 0) {
    Serial.println("recordWAV() returned null/empty");
    return false;
  }
  File f = SD.open(path, FILE_WRITE);
  if (!f) {
    free(wav_buffer);
    Serial.println("SD open failed for write");
    return false;
  }
  bool ok = (f.write(wav_buffer, wav_size) == wav_size);
  f.close();
  free(wav_buffer);
  if (!ok) Serial.println("SD write incomplete");
  return ok;
}

// ======================================================
// LOAD A CLIP FROM SD INTO myAudioBuffer
// ======================================================
bool myLoadClipFromSD(const char* path) {
  File f = SD.open(path);
  if (!f) {
    Serial.printf("  SD.open FAILED: %s\n", path);
    return false;
  }
  f.seek(44);   // skip WAV header
  size_t want = CLIP_SAMPLES * sizeof(int16_t);
  size_t got  = f.read((uint8_t*)myAudioBuffer, want);
  f.close();
  if (got < want) memset((uint8_t*)myAudioBuffer + got, 0, want - got);
  return (got > 0);
}

// ======================================================
// MEL FILTERBANK (built once, cached in PSRAM)
// ======================================================
static float myHzToMel(float hz)  { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float myMelToHz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

static float* myMelFbLo  = nullptr;
static float* myMelFbCtr = nullptr;
static float* myMelFbHi  = nullptr;

void myBuildMelFilterbank() {
  if (myMelFbCtr) return;
  myMelFbLo  = (float*)ps_malloc(MEL_BINS * sizeof(float));
  myMelFbCtr = (float*)ps_malloc(MEL_BINS * sizeof(float));
  myMelFbHi  = (float*)ps_malloc(MEL_BINS * sizeof(float));
  if (!myMelFbLo || !myMelFbCtr || !myMelFbHi) {
    Serial.println("FATAL: mel filterbank alloc failed");
    while (1) delay(1000);
  }
  float melLow   = myHzToMel(20.0f);
  float melHigh  = myHzToMel(SAMPLE_RATE / 2.0f);
  float melStep  = (melHigh - melLow) / (MEL_BINS + 1);
  float binScale = (float)FFT_SIZE / (float)SAMPLE_RATE;
  for (int m = 0; m < MEL_BINS; m++) {
    myMelFbLo[m]  = myMelToHz(melLow + (m    ) * melStep) * binScale;
    myMelFbCtr[m] = myMelToHz(melLow + (m + 1) * melStep) * binScale;
    myMelFbHi[m]  = myMelToHz(melLow + (m + 2) * melStep) * binScale;
  }
  Serial.println("Mel filterbank built");
}

// ======================================================
// POWER SPECTRUM — inline radix-2 Cooley-Tukey FFT  O(N log N)
//
// Replaces the O(N²) DFT from v10.  For FFT_SIZE=512:
//   Old DFT : 512 × 257 × 2 trig calls × 32 frames ≈ 8.4 M trig calls/clip
//   New FFT : 512 × log2(512) × 32 frames            ≈ 147 K trig calls/clip
// Measured speedup on ESP32-S3: ~35× per clip, from ~1.5 s to ~45 ms.
//
// Requirements: n must be a power of 2 (FFT_SIZE=512 satisfies this).
// In-place bit-reversal + butterfly on separate real/imag scratch arrays.
// myFftReal / myFftImag are reused across calls (allocated in PSRAM).
// vTaskDelay(0) is still called once per frame to keep the WDT happy.
// ======================================================

// Scratch arrays — allocated once alongside the other PSRAM buffers.
// Declared here as extern; definition lives in myAllocateMemory().
static float* myFftReal = nullptr;
static float* myFftImag = nullptr;

// Precomputed twiddle table (cos/sin) for FFT_SIZE/2 factors, built once.
static float* myTwiddleCos = nullptr;
static float* myTwiddleSin = nullptr;

void myBuildTwiddleTable() {
  if (myTwiddleCos) return;  // already built
  myTwiddleCos = (float*)ps_malloc((FFT_SIZE / 2) * sizeof(float));
  myTwiddleSin = (float*)ps_malloc((FFT_SIZE / 2) * sizeof(float));
  if (!myTwiddleCos || !myTwiddleSin) {
    Serial.println("FATAL: twiddle table alloc failed");
    while (1) delay(1000);
  }
  for (int k = 0; k < FFT_SIZE / 2; k++) {
    float angle = -2.0f * (float)M_PI * k / FFT_SIZE;
    myTwiddleCos[k] = cosf(angle);
    myTwiddleSin[k] = sinf(angle);
  }
}

// Compute power spectrum of `frame` (length n, must be power-of-2).
// Result written to `power[0..n/2]`  (n/2+1 bins).
// Uses myFftReal / myFftImag as scratch; caller must ensure they are allocated.
static void myPowerSpectrum(const float* frame, float* power, int n) {
  // Copy input into real array, zero imaginary
  for (int i = 0; i < n; i++) { myFftReal[i] = frame[i]; myFftImag[i] = 0.0f; }

  // --- Bit-reversal permutation ---
  int half = n >> 1;
  for (int i = 1, j = 0; i < n; i++) {
    int bit = half;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = myFftReal[i]; myFftReal[i] = myFftReal[j]; myFftReal[j] = tr;
      float ti = myFftImag[i]; myFftImag[i] = myFftImag[j]; myFftImag[j] = ti;
    }
  }

  // --- Cooley-Tukey butterfly stages ---
  for (int len = 2; len <= n; len <<= 1) {
    int half_len = len >> 1;
    int step = n / len;   // step into twiddle table
    for (int i = 0; i < n; i += len) {
      for (int k = 0; k < half_len; k++) {
        int tw = k * step;
        float wr = myTwiddleCos[tw];
        float wi = myTwiddleSin[tw];
        float ur = myFftReal[i + k];
        float ui = myFftImag[i + k];
        float vr = myFftReal[i + k + half_len] * wr - myFftImag[i + k + half_len] * wi;
        float vi = myFftReal[i + k + half_len] * wi + myFftImag[i + k + half_len] * wr;
        myFftReal[i + k]           = ur + vr;
        myFftImag[i + k]           = ui + vi;
        myFftReal[i + k + half_len] = ur - vr;
        myFftImag[i + k + half_len] = ui - vi;
      }
    }
  }

  // --- Power spectrum (magnitude squared) for bins 0..n/2 ---
  int bins = n / 2 + 1;
  for (int k = 0; k < bins; k++)
    power[k] = myFftReal[k] * myFftReal[k] + myFftImag[k] * myFftImag[k];
}

// ======================================================
// SPECTROGRAM  (myAudioBuffer -> myInputBuffer, normalised 0-1)
//
// v8 normalization fix:
//   Old code tracked globalMax initialised to 1e-8. Since real audio
//   log10 energies are negative (-8 to -1), globalMax stayed at ~0 and
//   the normalization range was always ~8, regardless of actual clip
//   content. Silence clips have a true max well below 0, so almost all
//   values were mapped to ≈1.0 — constant grey images.
//   Fix: two-pass — first pass stores raw log energies and finds the
//   true per-clip [min, max]; second pass normalises to [0, 1].
//   Range is floored at 1.0 so a pure-silence clip still produces a
//   meaningful (all-zero) image rather than dividing by ~0.
// ======================================================
bool myComputeSpectrogram() {
  if (!myAudioBuffer || !myInputBuffer || !myMelFbCtr) return false;
  if (!mySpectroFrame || !mySpectroPower)              return false;

  // Pass 1: compute raw log-mel energies into myInputBuffer
  float clipMin =  1e9f;
  float clipMax = -1e9f;

  for (int fr = 0; fr < MEL_FRAMES; fr++) {
    int startSample = fr * FRAME_SIZE;
    memset(mySpectroFrame, 0, FFT_SIZE * sizeof(float));
    for (int i = 0; i < FRAME_SIZE && i < FFT_SIZE; i++) {
      float hann = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (FRAME_SIZE - 1)));
      mySpectroFrame[i] = (myAudioBuffer[startSample + i] / 32768.0f) * hann;
    }
    myPowerSpectrum(mySpectroFrame, mySpectroPower, FFT_SIZE);

    for (int m = 0; m < MEL_BINS; m++) {
      int loB  = (int)myMelFbLo[m];
      int ctrB = (int)myMelFbCtr[m];
      int hiB  = (int)min((float)(FFT_SIZE / 2), myMelFbHi[m]);
      float energy = 0;
      for (int b = loB; b < ctrB && b <= FFT_SIZE / 2; b++) {
        float w = (b - myMelFbLo[m]) / (myMelFbCtr[m] - myMelFbLo[m] + 1e-8f);
        energy += w * mySpectroPower[b];
      }
      for (int b = ctrB; b < hiB && b <= FFT_SIZE / 2; b++) {
        float w = (myMelFbHi[m] - b) / (myMelFbHi[m] - myMelFbCtr[m] + 1e-8f);
        energy += w * mySpectroPower[b];
      }
      float logE = log10f(energy + 1e-6f);   // +1e-6 avoids -inf on silence
      myInputBuffer[fr * MEL_BINS + m] = logE;
      if (logE < clipMin) clipMin = logE;
      if (logE > clipMax) clipMax = logE;
    }

    // With the FFT the spectrogram is ~35x faster; vTaskDelay(0) after
    // each frame still keeps the WDT happy with negligible overhead.
    vTaskDelay(0);
  }

  // Pass 2: normalise using true per-clip range
  float range = clipMax - clipMin;
  if (range < 1.0f) range = 1.0f;   // floor: silence clips map to all-zero
  for (int i = 0; i < MEL_FRAMES * MEL_BINS; i++)
    myInputBuffer[i] = constrain((myInputBuffer[i] - clipMin) / range, 0.0f, 1.0f);

  return true;
}

bool myLoadClipAndComputeSpectrogram(const char* path) {
  return myLoadClipFromSD(path) && myComputeSpectrogram();
}

// ======================================================
// MEMORY ALLOCATION (PSRAM) — every allocation checked
// ======================================================
// Helper macro so failures name themselves
#define MY_PSRAM_MALLOC(ptr, type, count, label)                        \
  ptr = (type*)ps_malloc((count) * sizeof(type));                       \
  if (!ptr) { Serial.println("PSRAM fail: " label); myPsramFailed=true; }

#define MY_PSRAM_CALLOC(ptr, type, count, label)                        \
  ptr = (type*)ps_calloc((count), sizeof(type));                        \
  if (!ptr) { Serial.println("PSRAM fail: " label); myPsramFailed=true; }

void myAllocateMemory() {
  if (myInputBuffer) return;
  Serial.println("\n=== Allocating PSRAM ===");
  bool myPsramFailed = false;

  MY_PSRAM_MALLOC(myAudioBuffer,  int16_t, CLIP_SAMPLES,           "audioBuffer")
  MY_PSRAM_MALLOC(myInputBuffer,  float,   MEL_FRAMES * MEL_BINS,  "inputBuffer")
  MY_PSRAM_MALLOC(mySpectroFrame, float,   FFT_SIZE,               "spectroFrame")
  MY_PSRAM_MALLOC(mySpectroPower, float,   FFT_SIZE / 2 + 1,       "spectroPower")
  MY_PSRAM_MALLOC(myFftReal,      float,   FFT_SIZE,               "fftReal")      // FFT scratch
  MY_PSRAM_MALLOC(myFftImag,      float,   FFT_SIZE,               "fftImag")      // FFT scratch

  MY_PSRAM_MALLOC(myConv1_w,  float, CONV1_WEIGHTS,  "conv1_w")
  MY_PSRAM_MALLOC(myConv1_b,  float, CONV1_FILTERS,  "conv1_b")
  MY_PSRAM_MALLOC(myConv2_w,  float, CONV2_WEIGHTS,  "conv2_w")
  MY_PSRAM_MALLOC(myConv2_b,  float, CONV2_FILTERS,  "conv2_b")
  MY_PSRAM_MALLOC(myOutput_w, float, OUTPUT_WEIGHTS,  "output_w")
  MY_PSRAM_MALLOC(myOutput_b, float, NUM_CLASSES,     "output_b")

  MY_PSRAM_MALLOC(myConv1_w_grad,  float, CONV1_WEIGHTS,  "conv1_w_grad")
  MY_PSRAM_MALLOC(myConv1_b_grad,  float, CONV1_FILTERS,  "conv1_b_grad")
  MY_PSRAM_MALLOC(myConv2_w_grad,  float, CONV2_WEIGHTS,  "conv2_w_grad")
  MY_PSRAM_MALLOC(myConv2_b_grad,  float, CONV2_FILTERS,  "conv2_b_grad")
  MY_PSRAM_MALLOC(myOutput_w_grad, float, OUTPUT_WEIGHTS,  "output_w_grad")
  MY_PSRAM_MALLOC(myOutput_b_grad, float, NUM_CLASSES,     "output_b_grad")

  MY_PSRAM_CALLOC(myConv1_w_m,  float, CONV1_WEIGHTS,  "conv1_w_m")
  MY_PSRAM_CALLOC(myConv1_w_v,  float, CONV1_WEIGHTS,  "conv1_w_v")
  MY_PSRAM_CALLOC(myConv1_b_m,  float, CONV1_FILTERS,  "conv1_b_m")
  MY_PSRAM_CALLOC(myConv1_b_v,  float, CONV1_FILTERS,  "conv1_b_v")
  MY_PSRAM_CALLOC(myConv2_w_m,  float, CONV2_WEIGHTS,  "conv2_w_m")
  MY_PSRAM_CALLOC(myConv2_w_v,  float, CONV2_WEIGHTS,  "conv2_w_v")
  MY_PSRAM_CALLOC(myConv2_b_m,  float, CONV2_FILTERS,  "conv2_b_m")
  MY_PSRAM_CALLOC(myConv2_b_v,  float, CONV2_FILTERS,  "conv2_b_v")
  MY_PSRAM_CALLOC(myOutput_w_m, float, OUTPUT_WEIGHTS,  "output_w_m")
  MY_PSRAM_CALLOC(myOutput_w_v, float, OUTPUT_WEIGHTS,  "output_w_v")
  MY_PSRAM_CALLOC(myOutput_b_m, float, NUM_CLASSES,     "output_b_m")
  MY_PSRAM_CALLOC(myOutput_b_v, float, NUM_CLASSES,     "output_b_v")

  MY_PSRAM_MALLOC(myConv1_output, float, CONV1_OUT_H * CONV1_OUT_W * CONV1_FILTERS, "conv1_out")
  MY_PSRAM_MALLOC(myPool1_output, float, POOL1_OUT_H * POOL1_OUT_W * CONV1_FILTERS, "pool1_out")
  myPool1_argmax = (uint8_t*)ps_malloc(POOL1_OUT_H * POOL1_OUT_W * CONV1_FILTERS * sizeof(uint8_t));
  if (!myPool1_argmax) { Serial.println("PSRAM fail: pool1_argmax"); myPsramFailed = true; }
  MY_PSRAM_MALLOC(myConv2_output, float, CONV2_OUT_H * CONV2_OUT_W * CONV2_FILTERS, "conv2_out")
  MY_PSRAM_MALLOC(myDense_output, float, NUM_CLASSES,  "dense_out")

  MY_PSRAM_MALLOC(myDense_grad, float, FLATTENED_SIZE,                              "dense_grad")
  MY_PSRAM_MALLOC(myConv2_grad, float, CONV2_OUT_H * CONV2_OUT_W * CONV2_FILTERS,  "conv2_grad")
  MY_PSRAM_MALLOC(myPool1_grad, float, POOL1_OUT_H * POOL1_OUT_W * CONV1_FILTERS,  "pool1_grad")
  MY_PSRAM_MALLOC(myConv1_grad, float, CONV1_OUT_H * CONV1_OUT_W * CONV1_FILTERS,  "conv1_grad")

  if (myPsramFailed) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while (1) delay(1000);
  }
  Serial.printf("Free PSRAM after alloc: %d bytes\n", ESP.getFreePsram());

  // He initialisation
  float c1std = sqrtf(2.0f / (9.0f * INPUT_CH));
  for (int i = 0; i < CONV1_WEIGHTS; i++)
    myConv1_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c1std;
  for (int i = 0; i < CONV1_FILTERS; i++) myConv1_b[i] = 0;

  float c2std = sqrtf(2.0f / (9.0f * CONV1_FILTERS));
  for (int i = 0; i < CONV2_WEIGHTS; i++)
    myConv2_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c2std;
  for (int i = 0; i < CONV2_FILTERS; i++) myConv2_b[i] = 0;

  float dstd = sqrtf(2.0f / FLATTENED_SIZE);
  for (int i = 0; i < OUTPUT_WEIGHTS; i++)
    myOutput_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * dstd;
  for (int i = 0; i < NUM_CLASSES; i++) myOutput_b[i] = 0;

  Serial.println("He-init complete");
}

// ======================================================
// WEIGHT SAVE / LOAD / EXPORT
// ======================================================
void myExportHeader() {
  if (!mySDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MODEL_H\n#define MY_MODEL_H");
  file.printf("// NUM_CLASSES %d\n// Labels:", NUM_CLASSES);
  for (int i = 0; i < NUM_CLASSES; i++) file.printf(" \"%s\"", myClassLabels[i].c_str());
  file.println("\n// Uncomment #define USE_BAKED_WEIGHTS in main sketch to use these.");
  auto myDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = {", name);
    for (int i = 0; i < size; i++) {
      file.print(data[i], 6); file.print("f");
      if (i < size - 1) file.print(",");
      if ((i + 1) % 8 == 0) file.println();
    }
    file.println("};");
  };
  myDump("myModel_conv1_w",  myConv1_w,  CONV1_WEIGHTS);
  myDump("myModel_conv1_b",  myConv1_b,  CONV1_FILTERS);
  myDump("myModel_conv2_w",  myConv2_w,  CONV2_WEIGHTS);
  myDump("myModel_conv2_b",  myConv2_b,  CONV2_FILTERS);
  myDump("myModel_output_w", myOutput_w, OUTPUT_WEIGHTS);
  myDump("myModel_output_b", myOutput_b, NUM_CLASSES);
  file.println("#endif");
  file.close();
  Serial.println("Header exported: /header/myWeights.h");
}

bool myLoadWeights() {
  if (!mySDavailable || !SD.exists("/header/myWeights.bin")) return false;
  File f = SD.open("/header/myWeights.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
  f.read((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
  f.read((uint8_t*)myConv2_w,  CONV2_WEIGHTS  * 4);
  f.read((uint8_t*)myConv2_b,  CONV2_FILTERS  * 4);
  f.read((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
  f.read((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
  f.close();
  myWeightsTrained = true;
  Serial.println("Weights loaded from SD");
  return true;
}

void mySaveWeights() {
  if (!mySDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
    f.write((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
    f.write((uint8_t*)myConv2_w,  CONV2_WEIGHTS  * 4);
    f.write((uint8_t*)myConv2_b,  CONV2_FILTERS  * 4);
    f.write((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
    f.write((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
    f.close();
    Serial.println("Weights saved");
  }
  myExportHeader();
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  FORWARD / BACKWARD PASS + OPTIMIZER                                      ██
// ██████████████████████████████████████████████████████████████████████████████

void myForwardPass(float* input, float* logits) {
  // Conv1 (1-channel input)
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUT_H * CONV1_OUT_W;
    for (int y = 0; y < CONV1_OUT_H; y++) {
      for (int x = 0; x < CONV1_OUT_W; x++) {
        float sum = 0;
        for (int ky = 0; ky < 3; ky++)
          for (int kx = 0; kx < 3; kx++)
            sum += input[(y + ky) * INPUT_W + (x + kx)] * myConv1_w[f * 9 + ky * 3 + kx];
        myConv1_output[ob + y * CONV1_OUT_W + x] = leaky_relu(clip_value(sum + myConv1_b[f]));
      }
    }
  }
  // Pool1 (2x2 max) — records which cell was max in myPool1_argmax
  // Offsets: 0=(iy,ix)  1=(iy,ix+1)  2=(iy+1,ix)  3=(iy+1,ix+1)
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ib = f * CONV1_OUT_H * CONV1_OUT_W;
    int ob = f * POOL1_OUT_H * POOL1_OUT_W;
    for (int y = 0; y < POOL1_OUT_H; y++) {
      for (int x = 0; x < POOL1_OUT_W; x++) {
        int iy = y * 2, ix = x * 2;
        float v0 = myConv1_output[ib +  iy      * CONV1_OUT_W + ix    ];
        float v1 = myConv1_output[ib +  iy      * CONV1_OUT_W + ix + 1];
        float v2 = myConv1_output[ib + (iy + 1) * CONV1_OUT_W + ix    ];
        float v3 = myConv1_output[ib + (iy + 1) * CONV1_OUT_W + ix + 1];
        float mv = v0; uint8_t argmax = 0;
        if (v1 > mv) { mv = v1; argmax = 1; }
        if (v2 > mv) { mv = v2; argmax = 2; }
        if (v3 > mv) { mv = v3; argmax = 3; }
        myPool1_output[ob + y * POOL1_OUT_W + x] = mv;
        myPool1_argmax[ob + y * POOL1_OUT_W + x] = argmax;
      }
    }
  }
  // Conv2
  for (int f = 0; f < CONV2_FILTERS; f++) {
    int ob = f * CONV2_OUT_H * CONV2_OUT_W;
    for (int y = 0; y < CONV2_OUT_H; y++) {
      for (int x = 0; x < CONV2_OUT_W; x++) {
        float sum = 0;
        for (int c = 0; c < CONV1_FILTERS; c++) {
          int ib = c * POOL1_OUT_H * POOL1_OUT_W;
          for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
              sum += myPool1_output[ib + (y + ky) * POOL1_OUT_W + (x + kx)] *
                     myConv2_w[f * (CONV1_FILTERS * 9) + c * 9 + ky * 3 + kx];
        }
        myConv2_output[ob + y * CONV2_OUT_W + x] = leaky_relu(clip_value(sum + myConv2_b[f]));
      }
    }
  }
  // Dense + Softmax (Kahan summation for numerical stability)
  for (int c = 0; c < NUM_CLASSES; c++) {
    double sum = 0, comp = 0;
    for (int i = 0; i < FLATTENED_SIZE; i++) {
      double term = myConv2_output[i] * myOutput_w[c * FLATTENED_SIZE + i];
      double yy = term - comp; double t = sum + yy;
      comp = (t - sum) - yy; sum = t;
    }
    myDense_output[c] = clip_value((float)sum + myOutput_b[c], -50, 50);
  }
  float mx = myDense_output[0];
  for (int i = 1; i < NUM_CLASSES; i++) mx = max(mx, myDense_output[i]);
  float expSum = 0;
  for (int i = 0; i < NUM_CLASSES; i++) expSum += expf(myDense_output[i] - mx);
  for (int i = 0; i < NUM_CLASSES; i++) {
    logits[i]         = myDense_output[i];
    myDense_output[i] = expf(myDense_output[i] - mx) / expSum;
  }
}

void myBackwardDense(int label) {
  memset(myDense_grad, 0, FLATTENED_SIZE * sizeof(float));
  for (int c = 0; c < NUM_CLASSES; c++) {
    float err = myDense_output[c] - (c == label ? 1.0f : 0.0f);
    for (int i = 0; i < FLATTENED_SIZE; i++) {
      myOutput_w_grad[c * FLATTENED_SIZE + i] += err * myConv2_output[i];
      myDense_grad[i] += err * myOutput_w[c * FLATTENED_SIZE + i];
    }
    myOutput_b_grad[c] += err;
  }
}

void myBackwardConv2() {

/*
  for (int i = 0; i < FLATTENED_SIZE; i++)
    myConv2_grad[i] = myDense_grad[i] * leaky_relu_deriv(myConv2_output[i]);
*/
for (int f = 0; f < CONV2_FILTERS; f++) {
    int base = f * CONV2_OUT_H * CONV2_OUT_W;
    for (int y = 0; y < CONV2_OUT_H; y++) {
        for (int x = 0; x < CONV2_OUT_W; x++) {
            int idx = base + y * CONV2_OUT_W + x;
            myConv2_grad[idx] =
                myDense_grad[idx] * leaky_relu_deriv(myConv2_output[idx]);
        }
    }
}


  memset(myPool1_grad, 0, POOL1_OUT_H * POOL1_OUT_W * CONV1_FILTERS * sizeof(float));
  for (int f = 0; f < CONV2_FILTERS; f++) {
    int ob = f * CONV2_OUT_H * CONV2_OUT_W;
    for (int y = 0; y < CONV2_OUT_H; y++) {
      for (int x = 0; x < CONV2_OUT_W; x++) {
        float grad = myConv2_grad[ob + y * CONV2_OUT_W + x];
        myConv2_b_grad[f] += grad;
        for (int c = 0; c < CONV1_FILTERS; c++) {
          int ib = c * POOL1_OUT_H * POOL1_OUT_W;
          for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
              int pi = ib + (y + ky) * POOL1_OUT_W + (x + kx);
              int wi = f * (CONV1_FILTERS * 9) + c * 9 + ky * 3 + kx;
              myConv2_w_grad[wi] += grad * myPool1_output[pi];
              myPool1_grad[pi]   += grad * myConv2_w[wi];
            }
          }
        }
      }
    }
  }
}

void myBackwardPool1() {
  // v8 fix: use myPool1_argmax recorded during the forward pass.
  // The old exact float equality (== poolVal) was fragile after leaky_relu
  // and silently zeroed most Conv1 gradients, killing early-layer learning.
  memset(myConv1_grad, 0, CONV1_OUT_H * CONV1_OUT_W * CONV1_FILTERS * sizeof(float));
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ib = f * CONV1_OUT_H * CONV1_OUT_W;
    int ob = f * POOL1_OUT_H * POOL1_OUT_W;
    for (int y = 0; y < POOL1_OUT_H; y++) {
      for (int x = 0; x < POOL1_OUT_W; x++) {
        float grad   = myPool1_grad  [ob + y * POOL1_OUT_W + x];
        uint8_t amax = myPool1_argmax[ob + y * POOL1_OUT_W + x];
        int iy = y * 2, ix = x * 2;
        // Route gradient to exactly the cell that was the max
        int srcOffset;
        switch (amax) {
          case 0: srcOffset = ib +  iy      * CONV1_OUT_W + ix;     break;
          case 1: srcOffset = ib +  iy      * CONV1_OUT_W + ix + 1; break;
          case 2: srcOffset = ib + (iy + 1) * CONV1_OUT_W + ix;     break;
          default:srcOffset = ib + (iy + 1) * CONV1_OUT_W + ix + 1; break;
        }
        myConv1_grad[srcOffset] += grad;
      }
    }
  }
}

void myBackwardConv1() {
  for (int i = 0; i < CONV1_OUT_H * CONV1_OUT_W * CONV1_FILTERS; i++)
    myConv1_grad[i] *= leaky_relu_deriv(myConv1_output[i]);
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUT_H * CONV1_OUT_W;
    for (int y = 0; y < CONV1_OUT_H; y++) {
      for (int x = 0; x < CONV1_OUT_W; x++) {
        float grad = myConv1_grad[ob + y * CONV1_OUT_W + x];
        myConv1_b_grad[f] += grad;
        for (int ky = 0; ky < 3; ky++)
          for (int kx = 0; kx < 3; kx++)
            myConv1_w_grad[f * 9 + ky * 3 + kx] +=
              grad * myInputBuffer[(y + ky) * INPUT_W + (x + kx)];
      }
    }
  }
}

// ======================================================
// ADAM OPTIMIZER — v7 fixes:
//   1. Accepts a gradient scale factor (1/processed) for batch normalization.
//   2. Precomputes bias-correction factor outside the per-weight loop.
//   3. Gradient clipping applied before m/v update.
// ======================================================
void myAdamUpdate(float* w, float* g, float* m, float* v,
                  int size, int step, float gradScale) {
  const float b1  = 0.9f, b2 = 0.999f, eps = 1e-6f;
  // Precompute bias-corrected learning rate for this step
  float b1pow  = powf(b1, step);
  float b2pow  = powf(b2, step);
  float lr_t   = LEARNING_RATE * sqrtf(1.0f - b2pow) / (1.0f - b1pow + 1e-12f);

  for (int i = 0; i < size; i++) {
    float gi = g[i] * gradScale;                             // normalize by batch
    gi = constrain(gi, -1.0f, 1.0f);                         // gradient clip
    m[i] = b1 * m[i] + (1.0f - b1) * gi;
    v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
    w[i] -= lr_t * m[i] / (sqrtf(v[i]) + eps);
    w[i]  = clip_value(w[i], -10.0f, 10.0f);
  }
}

void myUpdateWeights(int step, float gradScale) {
  myAdamUpdate(myConv1_w,  myConv1_w_grad,  myConv1_w_m,  myConv1_w_v,  CONV1_WEIGHTS,  step, gradScale);
  myAdamUpdate(myConv1_b,  myConv1_b_grad,  myConv1_b_m,  myConv1_b_v,  CONV1_FILTERS,  step, gradScale);
  myAdamUpdate(myConv2_w,  myConv2_w_grad,  myConv2_w_m,  myConv2_w_v,  CONV2_WEIGHTS,  step, gradScale);
  myAdamUpdate(myConv2_b,  myConv2_b_grad,  myConv2_b_m,  myConv2_b_v,  CONV2_FILTERS,  step, gradScale);
  myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, OUTPUT_WEIGHTS, step, gradScale);
  myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, NUM_CLASSES,    step, gradScale);
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  MENU SYSTEM                                                               ██
// ██████████████████████████████████████████████████████████████████████████████

void myResetMenuState();
void myDrawMenu();
void myActionCollect(int classIdx);
void myActionTrain();
void myActionInfer();

void myResetMenuState() {
  myIsSelected = false;
  myResetTouchState();
  myLastActivityTime = millis();
  myDrawMenu();
}

void myDrawMenu() {
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    String label = (i <= NUM_CLASSES) ? myClassLabels[i - 1] :
                   (i == NUM_CLASSES + 1) ? "Train" : "Infer";
    Serial.printf("%s %d. %s\n", i == myMenuIndex ? " >" : "  ", i, label.c_str());
  }
  Serial.println("t=next  l=select  1-9=direct");

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");
    int start = (myMenuIndex <= NUM_CLASSES) ? 1 : myMenuIndex - 2;
    for (int i = 0; i < 3; i++) {
      int cur = start + i;
      if (cur > myTotalItems) break;
      String label = (cur <= NUM_CLASSES) ? myClassLabels[cur - 1] :
                     (cur == NUM_CLASSES + 1) ? "Train" : "Infer";
      u8g2.drawStr(0, 18 + i * 9,
                   (cur == myMenuIndex ? "> " + label : "  " + label).c_str());
    }
  } while (u8g2.nextPage());
}

void myExecuteMenuItem(int idx) {
  if (idx <= NUM_CLASSES)           myActionCollect(idx - 1);
  else if (idx == NUM_CLASSES + 1)  myActionTrain();
  else                              myActionInfer();
}

void myHandleMenuNavigation() {
  unsigned long now = millis();
  if (!myIsSelected && Serial.available()) {
    char c = Serial.read();
    if (c >= '1' && c <= '9') {
      int n = c - '0';
      if (n <= myTotalItems) {
        myMenuIndex = n; myIsSelected = true;
        myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
      }
    } else if (c == 't' || c == 'T') {
      if (now - myLastTapTime > myTapCooldown) {
        myMenuIndex = (myMenuIndex % myTotalItems) + 1;
        myDrawMenu(); myLastTapTime = now; myLastActivityTime = now;
      }
    } else if (c == 'l' || c == 'L') {
      myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
    }
  }
  if (!myIsSelected) {
    int ta = myCheckTouchInput();
    if (ta == 1) {
      if (now - myLastTapTime > myTapCooldown) {
        myMenuIndex = (myMenuIndex % myTotalItems) + 1;
        myDrawMenu(); myLastTapTime = now; myLastActivityTime = now;
      }
    } else if (ta == 2) {
      myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
    }
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  SETUP & LOOP                                                              ██
// ██████████████████████████████████████████████████████████████████████████████

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(500);
  Serial.println("\n=== XIAO ESP32-S3 Audio ML v7 ===");
  Serial.printf("Heap: %d  PSRAM: %d\n", ESP.getFreeHeap(), ESP.getFreePsram());

  pinMode(A0, INPUT);
  u8g2.begin();

  // SD card
  pinMode(21, OUTPUT); digitalWrite(21, HIGH); delay(100);
  SPI.begin();
  mySDavailable = SD.begin(21, SPI, 16000000, "/sd", 5, false);  // 16 MHz data speed
  if (!mySDavailable) {
    SD.end();
    Serial.println("No SD card");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
  } else {
    Serial.println("SD mounted");
  }

  // I2S mic
  myI2S.setPinsPdmRx(42, 41);
  if (!myI2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("FATAL: I2S init failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "MIC FAIL!"); } while (u8g2.nextPage());
    while (1) delay(1000);
  }
  Serial.println("I2S mic OK");

  myAllocateMemory();
  myBuildMelFilterbank();
  myBuildTwiddleTable();

#ifdef USE_BAKED_WEIGHTS
  memcpy(myConv1_w,  myModel_conv1_w,  CONV1_WEIGHTS  * sizeof(float));
  memcpy(myConv1_b,  myModel_conv1_b,  CONV1_FILTERS  * sizeof(float));
  memcpy(myConv2_w,  myModel_conv2_w,  CONV2_WEIGHTS  * sizeof(float));
  memcpy(myConv2_b,  myModel_conv2_b,  CONV2_FILTERS  * sizeof(float));
  memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
  memcpy(myOutput_b, myModel_output_b, NUM_CLASSES    * sizeof(float));
  myWeightsTrained = true;
  Serial.println("Baked-in weights loaded");
#endif

  if (myLoadWeights()) Serial.println("SD weights loaded — inference ready");

  myResetMenuState();
  delay(1000);
  Serial.println("Ready. Tap A0 to navigate, 3+ taps to select.");
  myDrawMenu();
}

void loop() {
  myHandleMenuNavigation();
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 1: AUDIO COLLECTION                                                  ██
// ██████████████████████████████████████████████████████████████████████████████

void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  Serial.printf("\n>>> Collect: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("  TAP=record  LONG PRESS=exit  L=exit");
  myResetTouchState();

  String folderPath = "/audio/" + myClassLabels[classIdx];
  if (!SD.exists("/audio"))   SD.mkdir("/audio");
  if (!SD.exists(folderPath)) SD.mkdir(folderPath);

  // Count existing clips
  int clipCount = 0;
  {
    File root = SD.open(folderPath);
    if (root) {
      while (File f = root.openNextFile()) {
        if (!f.isDirectory() && String(f.name()).endsWith(".wav")) clipCount++;
        f.close();
      }
      root.close();
    }
  }

  auto myShowIdle = [&]() {
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 10, myClassLabels[classIdx].c_str());
      char buf[20]; snprintf(buf, sizeof(buf), "Clips: %d", clipCount);
      u8g2.drawStr(0, 22, buf);
      u8g2.drawStr(0, 34, "TAP to record");
    } while (u8g2.nextPage());
  };
  myShowIdle();

  bool shouldRecord = false;

  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if      (c == 'l' || c == 'L') { myResetMenuState(); return; }
      else if (c == 't' || c == 'T') shouldRecord = true;
    }
    int ta = myCheckTouchInput();
    if      (ta == 2) { myResetMenuState(); return; }
    else if (ta == 1) shouldRecord = true;

    if (shouldRecord) {
      shouldRecord = false;

      u8g2.firstPage();
      do {
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(0, 12, "GET READY...");
        u8g2.drawStr(0, 26, "Recording in 1s");
      } while (u8g2.nextPage());
      delay(800);

      u8g2.firstPage();
      do {
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(0, 15, "** RECORDING **");
        u8g2.drawStr(0, 28, "1 second...");
      } while (u8g2.nextPage());

      String fileName = folderPath + "/clip_" + String(millis()) + ".wav";
      if (mySaveClipToSD(fileName)) {
        clipCount++;
        Serial.printf("Saved: %s  (total: %d)\n", fileName.c_str(), clipCount);
      } else {
        Serial.println("ERROR: save failed");
        u8g2.firstPage();
        do { u8g2.drawStr(0, 15, "SAVE FAILED"); } while (u8g2.nextPage());
        delay(1000);
      }

      myShowIdle();
    }
    delay(10);
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 2: TRAINING                                                          ██
// ██████████████████████████████████████████████████████████████████████████████
//
// v7 key changes vs v6:
//   - Gradient scale = 1 / processed clips per batch (fixes the biggest bug).
//   - Adam step counter increments only on actual updates (not batch index).
//   - Serial prints every epoch: Ep N/20  Loss:x.xx  Acc:xx%  Val:xx%
//   - OLED updates every epoch showing: epoch/total, loss, acc, free PSRAM.
//   - OLED also shows a mini progress bar for the full training run.
//   - Validation split skips hold-out if class has too few clips.
//   - NaN/Inf check after loss; corrupt samples skipped cleanly.
//   - Watchdog yields inside clip loading loop.

void myActionTrain() {
  if (!mySDavailable) {
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  Serial.println("\n>>> Training  (L or 3+ taps = save & exit)");
  myResetTouchState();

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 15, "TRAINING MODE");
    u8g2.drawStr(0, 28, "Scanning SD...");
  } while (u8g2.nextPage());

  if (myLoadWeights()) Serial.println("Continuing from saved weights");
  else                 Serial.println("Starting fresh (He-init)");

  while (true) {
    // ---- Scan SD — build FULL paths explicitly ----
    myTrainingData.clear();
    int classCounts[NUM_CLASSES] = {};

    for (int i = 0; i < NUM_CLASSES; i++) {
      String folderPath = "/audio/" + myClassLabels[i];
      File root = SD.open(folderPath);
      if (root) {
        while (File f = root.openNextFile()) {
          if (!f.isDirectory() && String(f.name()).endsWith(".wav")) {
            String fname = String(f.name());
            if (fname.startsWith("/")) fname = fname.substring(fname.lastIndexOf("/") + 1);
            String fullPath = folderPath + "/" + fname;
            myTrainingData.push_back({fullPath, i});
            classCounts[i]++;
          }
          f.close();
        }
        root.close();
      }
    }

    // Report scan results
    Serial.println("--- Clip scan ---");
    for (int i = 0; i < NUM_CLASSES; i++)
      Serial.printf("  %s: %d clips\n", myClassLabels[i].c_str(), classCounts[i]);

    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.drawStr(0, 8, "Clips found:");
      for (int i = 0; i < min(NUM_CLASSES, 3); i++) {
        char buf[24];
        snprintf(buf, sizeof(buf), " %s:%d", myClassLabels[i].c_str(), classCounts[i]);
        u8g2.drawStr(0, 16 + i * 9, buf);
      }
    } while (u8g2.nextPage());
    delay(1500);

    if (myTrainingData.empty()) {
      u8g2.firstPage();
      do { u8g2.drawStr(0, 20, "No clips found!"); } while (u8g2.nextPage());
      delay(2000); myResetMenuState(); return;
    }

    // Sort for deterministic validation split
    std::sort(myTrainingData.begin(), myTrainingData.end(),
              [](const TrainingItem& a, const TrainingItem& b) { return a.path < b.path; });

    // ---- Validation split — safe for small datasets ----
    // Each class keeps at least 1 clip for training.
    // If a class has <= VALIDATION_CLIPS clips, 0 are held out (warn).
    std::vector<TrainingItem> myValidationData;
    {
      int counts[NUM_CLASSES] = {};
      for (auto& item : myTrainingData) counts[item.label]++;

      int holdOut[NUM_CLASSES];
      for (int c = 0; c < NUM_CLASSES; c++) {
        if (counts[c] <= VALIDATION_CLIPS + 1) {
          holdOut[c] = 0;   // not enough clips — keep all for training
          Serial.printf("  WARN: %s has only %d clips, skipping val hold-out\n",
                        myClassLabels[c].c_str(), counts[c]);
        } else {
          holdOut[c] = VALIDATION_CLIPS;
        }
      }

      std::vector<TrainingItem> trainOnly;
      int seen[NUM_CLASSES] = {};
      for (int i = (int)myTrainingData.size() - 1; i >= 0; i--) {
        int c = myTrainingData[i].label;
        if (seen[c] < holdOut[c]) { myValidationData.push_back(myTrainingData[i]); seen[c]++; }
        else                       trainOnly.push_back(myTrainingData[i]);
      }
      myTrainingData = trainOnly;
    }

    int total           = (int)myTrainingData.size();
    int batchesPerEpoch = max(1, (total + BATCH_SIZE - 1) / BATCH_SIZE);

    // Safety check — can't train with nothing
    if (total == 0) {
      Serial.println("ERROR: No training clips after validation split!");
      u8g2.firstPage();
      do { u8g2.drawStr(0, 20, "No train clips!"); } while (u8g2.nextPage());
      delay(2000); myResetMenuState(); return;
    }

    Serial.printf("Train: %d  Val: %d  BatchSz: %d  Epochs: %d\n",
                  total, (int)myValidationData.size(), BATCH_SIZE, TARGET_EPOCHS);
    // Rough timing: each clip takes ~4 s for the O(N^2) spectrogram
    float secsPerEpoch = total * 0.05f;   // FFT: ~50 ms/clip vs ~1.5 s for old DFT
    Serial.printf("Estimated: ~%.0f s/epoch  (~%.0f min total)\n",
                  secsPerEpoch, secsPerEpoch * TARGET_EPOCHS / 60.0f);

    std::vector<int> indices;
    indices.reserve(total);
    for (int i = 0; i < total; i++) indices.push_back(i);

    int adamStep = 0;   // tracks actual gradient updates for Adam bias correction

    for (int epoch = 0; epoch < TARGET_EPOCHS; epoch++) {

      // Shuffle indices at epoch start
      for (int i = total - 1; i > 0; i--) {
        int j = random(i + 1);
        int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
      }

      // Print epoch header to serial
      Serial.printf("\n--- Epoch %d/%d ---\n", epoch + 1, TARGET_EPOCHS);
      Serial.printf("  Free heap: %d  Free PSRAM: %d\n",
                    ESP.getFreeHeap(), ESP.getFreePsram());

      float epochLoss   = 0;
      int   epochCorrect = 0;
      int   epochSamples = 0;
      int   loadFails    = 0;

      for (int b = 0; b < batchesPerEpoch; b++) {
        // Exit check
        if (Serial.available()) {
          char c = Serial.read();
          if (c == 'l' || c == 'L' || c == 'x' || c == 'X') {
            mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
          }
        }
        myCheckTouchBackground();
        if (myPeekTouchAction() == 2) {
          myCheckTouchInput();
          mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
        }

        int batchStart = b * BATCH_SIZE;
        int batchEnd   = min(batchStart + BATCH_SIZE, total);

        // Zero gradients
        memset(myConv1_w_grad,  0, CONV1_WEIGHTS  * sizeof(float));
        memset(myConv1_b_grad,  0, CONV1_FILTERS  * sizeof(float));
        memset(myConv2_w_grad,  0, CONV2_WEIGHTS  * sizeof(float));
        memset(myConv2_b_grad,  0, CONV2_FILTERS  * sizeof(float));
        memset(myOutput_w_grad, 0, OUTPUT_WEIGHTS * sizeof(float));
        memset(myOutput_b_grad, 0, NUM_CLASSES    * sizeof(float));

        int processed = 0;
        float batchLoss = 0;
        int   batchCorrect = 0;
        int   batchLoadFails = 0;

        for (int i = batchStart; i < batchEnd; i++) {
          TrainingItem& clip = myTrainingData[indices[i]];

          if (!myLoadClipAndComputeSpectrogram(clip.path.c_str())) {
            loadFails++; batchLoadFails++;
            Serial.printf("  LOAD FAIL [%d]: %s\n", i, clip.path.c_str());
            continue;
          }

          float logits[NUM_CLASSES];
          myForwardPass(myInputBuffer, logits);

          float prob = max(myDense_output[clip.label], 1e-7f);
          float sampleLoss = -logf(prob);

          // NaN/Inf guard — skip this sample
          if (isnan(sampleLoss) || isinf(sampleLoss)) {
            Serial.println("  WARN: NaN/Inf loss — skipping sample");
            continue;
          }

          batchLoss += sampleLoss;

          int pred = 0;
          for (int j = 1; j < NUM_CLASSES; j++)
            if (myDense_output[j] > myDense_output[pred]) pred = j;
          if (pred == clip.label) batchCorrect++;

          myBackwardDense(clip.label);
          myBackwardConv2();
          myBackwardPool1();
          myBackwardConv1();

          processed++;

          // Periodic exit check inside batch
          if (i % 3 == 0) {
            myCheckTouchBackground();
            if (myPeekTouchAction() == 2) {
              myCheckTouchInput();
              mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
            }
            if (Serial.available()) {
              char c = Serial.read();
              if (c == 'l' || c == 'L' || c == 'x' || c == 'X') {
                mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
              }
            }
          }

          yield();   // watchdog feed inside clip loop
        }

        if (processed > 0) {
          adamStep++;
          float gradScale = 1.0f / processed;   // normalize gradients by batch size
          myUpdateWeights(adamStep, gradScale);

          epochLoss    += batchLoss / processed;
          epochCorrect += batchCorrect;
          epochSamples += processed;
        }

        if (batchLoadFails > 0)
          Serial.printf("  Batch %d: %d load fail(s)\n", b + 1, batchLoadFails);
      }

      // ---- Per-epoch summary ----
      float avgLoss = epochSamples > 0 ? epochLoss / batchesPerEpoch : 0;
      float avgAcc  = epochSamples > 0 ? 100.0f * epochCorrect / epochSamples : 0;

      // Validation pass
      float valAcc = -1;
      if (!myValidationData.empty()) {
        int valCorrect = 0, valCount = 0;
        for (auto& vitem : myValidationData) {
          if (!myLoadClipAndComputeSpectrogram(vitem.path.c_str())) continue;
          float logits[NUM_CLASSES];
          myForwardPass(myInputBuffer, logits);
          int pred = 0;
          for (int j = 1; j < NUM_CLASSES; j++)
            if (myDense_output[j] > myDense_output[pred]) pred = j;
          if (pred == vitem.label) valCorrect++;
          valCount++;
          yield();
        }
        if (valCount > 0) valAcc = 100.0f * valCorrect / valCount;
      }

      // Serial epoch summary — always printed
      if (valAcc >= 0)
        Serial.printf("Ep %d/%d  Loss:%.4f  Acc:%.1f%%  Val:%.1f%%\n",
                      epoch + 1, TARGET_EPOCHS, avgLoss, avgAcc, valAcc);
      else
        Serial.printf("Ep %d/%d  Loss:%.4f  Acc:%.1f%%\n",
                      epoch + 1, TARGET_EPOCHS, avgLoss, avgAcc);

      // OLED epoch summary — always updated
      {
        int oledW  = u8g2.getDisplayWidth();   // 72
        int barFull = (int)((float)(epoch + 1) / TARGET_EPOCHS * oledW);

        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);

          // Line 1: Ep N/20
          char line1[24];
          snprintf(line1, sizeof(line1), "Ep%d/%d", epoch + 1, TARGET_EPOCHS);
          u8g2.drawStr(0, 8, line1);

          // Line 2: Loss
          char line2[24];
          snprintf(line2, sizeof(line2), "Loss:%.3f", avgLoss);
          u8g2.drawStr(0, 17, line2);

          // Line 3: Train acc  (+ val if available)
          char line3[24];
          if (valAcc >= 0)
            snprintf(line3, sizeof(line3), "A:%.0f%% V:%.0f%%", avgAcc, valAcc);
          else
            snprintf(line3, sizeof(line3), "Acc: %.1f%%", avgAcc);
          u8g2.drawStr(0, 26, line3);

          // Line 4: progress bar (full width, 3 px tall)
          u8g2.drawFrame(0, 30, oledW, 6);
          if (barFull > 0) u8g2.drawBox(0, 30, barFull, 6);

        } while (u8g2.nextPage());
      }
    } // end epoch loop

    Serial.println("\n--- Training complete ---");
    mySaveWeights();
    myWeightsTrained = true;

    u8g2.firstPage();
    do {
      u8g2.drawStr(0, 12, "DONE!");
      u8g2.drawStr(0, 24, "Tap: train again");
      u8g2.drawStr(0, 36, "3+:  exit");
    } while (u8g2.nextPage());

    myResetTouchState();
    Serial.println("T=train again  L=exit");
    while (true) {
      if (Serial.available()) {
        char c = Serial.read();
        if (c == 'l' || c == 'L' || c == 'x' || c == 'X') { myResetMenuState(); return; }
        else if (c == 't' || c == 'T') break;
      }
      int ta = myCheckTouchInput();
      if      (ta == 2) { myResetMenuState(); return; }
      else if (ta == 1) break;
      delay(10);
    }
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 3: INFERENCE (VAD-triggered)                                         ██
// ██████████████████████████████████████████████████████████████████████████████

#define myVadWindowMs  100
#define myVadThreshold 500
#define myVadHoldMs    150

void myActionInfer() {
  if (!myWeightsTrained) {
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 12, "No weights!");
      u8g2.drawStr(0, 24, "Train first");
    } while (u8g2.nextPage());
    delay(3000); myResetMenuState(); return;
  }

  Serial.println("\n>>> Inference (VAD)  T or L to exit");
  myResetTouchState();

  int   lastPred  = 0;
  float lastConf  = 0;
  bool  vadActive = false;
  unsigned long vadRiseTime = 0;

  const int VAD_SAMPLES = (SAMPLE_RATE * myVadWindowMs) / 1000;
  static int16_t vadBuf[1600];   // static — avoid stack allocation of 3200 bytes

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 12, "LISTENING...");
    u8g2.drawStr(0, 26, "Waiting 4 sound");
  } while (u8g2.nextPage());

  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 't' || c == 'T' || c == 'l' || c == 'L') { myResetMenuState(); return; }
    }
    if (myCheckTouchInput() != 0) { myResetMenuState(); return; }

    size_t got        = myI2S.readBytes((char*)vadBuf, VAD_SAMPLES * sizeof(int16_t));
    int    samplesGot = (int)(got / sizeof(int16_t));
    int64_t sumSq = 0;
    for (int i = 0; i < samplesGot; i++) sumSq += (int64_t)vadBuf[i] * vadBuf[i];
    float rms = samplesGot > 0 ? sqrtf((float)(sumSq / samplesGot)) : 0;

    if (rms > myVadThreshold) {
      if (!vadActive) { vadActive = true; vadRiseTime = millis(); }
      else if (millis() - vadRiseTime >= myVadHoldMs) {
        vadActive = false;
        //Serial.printf("VAD triggered (RMS=%.0f)\n", rms);  // I don't need this

        size_t   wav_size   = 0;
        uint8_t* wav_buffer = myI2S.recordWAV(CLIP_SECONDS, &wav_size);
        bool classified = false;

        if (wav_buffer && wav_size > 44) {
          size_t pcmBytes = min(wav_size - 44,
                                (size_t)(CLIP_SAMPLES * sizeof(int16_t)));
          memcpy(myAudioBuffer, wav_buffer + 44, pcmBytes);
          if (pcmBytes < CLIP_SAMPLES * sizeof(int16_t))
            memset((uint8_t*)myAudioBuffer + pcmBytes, 0,
                   CLIP_SAMPLES * sizeof(int16_t) - pcmBytes);
          free(wav_buffer);

          if (myComputeSpectrogram()) {
            float logits[NUM_CLASSES];
            myForwardPass(myInputBuffer, logits);
            lastPred = 0;
            for (int i = 1; i < NUM_CLASSES; i++)
              if (myDense_output[i] > myDense_output[lastPred]) lastPred = i;
            lastConf   = myDense_output[lastPred];
            classified = true;

            Serial.printf("=> %s (%.1f%%)", myClassLabels[lastPred].c_str(), lastConf * 100);
            for (int i = 0; i < NUM_CLASSES; i++)
              Serial.printf("  %s:%.0f%%", myClassLabels[i].c_str(), myDense_output[i] * 100);
            Serial.println();

            // ---- OLED: class name + percent (top), bar chart (bottom) ----
            {
              const char* lbl = myClassLabels[lastPred].c_str();
              if (lbl[0] >= '0' && lbl[0] <= '9') lbl++;   // skip numeric prefix

              int oW  = u8g2.getDisplayWidth();    // 72
              int oH  = u8g2.getDisplayHeight();   // 40
              int pct = (int)(lastConf * 100);

              u8g2.firstPage();
              do {
                // --- Top: class name + confidence percent ---
                u8g2.setFont(u8g2_font_6x10_tf);
                u8g2.drawStr(0, 10, lbl);

                char pctBuf[8];
                snprintf(pctBuf, sizeof(pctBuf), "%d%%", pct);
                u8g2.drawStr(0, 22, pctBuf);

                // --- Separator ---
                u8g2.drawHLine(0, 24, oW);

                // --- Bottom: mini bar per class ---
                u8g2.setFont(u8g2_font_4x6_tf);
                int barAreaTop = 26;
                int barH = (oH - barAreaTop) / NUM_CLASSES;
                if (barH < 4) barH = 4;

                for (int c = 0; c < NUM_CLASSES; c++) {
                  int y    = barAreaTop + c * barH;
                  int barW = (int)(myDense_output[c] * (oW - 1));
                  if (barW < 1) barW = 1;
                  if (c == lastPred) u8g2.drawBox(0, y, barW, barH - 1);
                  else               u8g2.drawFrame(0, y, barW, barH - 1);
                }
              } while (u8g2.nextPage());
              // No delay — return to listening loop immediately
            }
          }
        } else {
          if (wav_buffer) free(wav_buffer);
          Serial.println("recordWAV() failed in inference");
        }

        // Back to listening — show last result continuously
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_6x10_tf);
          if (classified) {
            const char* lbl = myClassLabels[lastPred].c_str();
            if (lbl[0] >= '0' && lbl[0] <= '9') lbl++;
            u8g2.drawStr(0, 12, lbl);
            char buf[8];
            snprintf(buf, sizeof(buf), "%d%%", (int)(lastConf * 100));
            u8g2.drawStr(0, 26, buf);
          } else {
            u8g2.drawStr(0, 12, "Listening...");
          }
        } while (u8g2.nextPage());
      }
    } else {
      vadActive = false;
    }

    delay(5);
  }
}
