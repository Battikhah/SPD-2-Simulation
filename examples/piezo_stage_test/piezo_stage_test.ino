// ESP32 Dev Module: MCP602 and piezo bench test. Serial Monitor: 115200 baud.
#include <Arduino.h>
#include <math.h>

#define NORMAL 0
#define WORKAROUND 1
#define REF_MODE NORMAL
#define ENABLE_PIN7_ADC 0  // GPIO35, 1k to pin 7; NORMAL mode only
#define ENABLE_PIN5_ADC 0  // GPIO33, 1k to pin 5 / divider node

const uint8_t SIGNAL_PIN = 34, PIN7_ADC = 35, PIN5_ADC = 33;
const unsigned SAMPLE_US = 1000, BASE_SAMPLES = 1000, NOISE_SAMPLES = 5000;
const unsigned TAP_SAMPLES = 10000, SETTLE_SAMPLES = 50, LOCKOUT_MS = 150;
const float BIAS_MIN = 1500, BIAS_MAX = 1800, HIGH_MIN = 1900, HIGH_MAX = 2150;
const float LOW_CHECK = 200, HIGH_CHECK = 3000, MATCH_MV = 30;
const float NOISE_PASS_MV = 10, TAP_MIN_MV = 30, TAP_NOISE_MULT = 5;
const float SETTLE_NOISE_MULT = 2, BASE_DRIFT_MV = 30;
const int CLIP_LOW = 50, CLIP_HIGH = 2050, MAX_SETTLE_MS = 500;
const int MIN_TAPS = 4, MAX_TAPS = 6;
const unsigned MAX_EVENTS = 64;

#if REF_MODE != NORMAL && REF_MODE != WORKAROUND
#error Invalid REF_MODE
#endif
#if REF_MODE == WORKAROUND && ENABLE_PIN7_ADC
#error GPIO35 pin 7 measurement is only valid in NORMAL mode
#endif

struct Result { const char *name; bool done, pass; String values, hint; };
void printResult(const Result &r);  // Declare after Result for Arduino's prototype generator.
Result bias = {"Bias", false, false, "", ""};
Result opamp = {"Op-amp", false, false, "", ""};
Result noise = {"Noise", false, false, "", ""};
Result tap = {"Tap", false, false, "", ""};
uint16_t samples[TAP_SAMPLES];
float noiseRms = 0;
bool haveNoise = false;
unsigned lateSamples = 0;

// Schedule from the preceding deadline; flag missed sample periods.
void capture(uint8_t pin, unsigned count) {
  uint32_t next = micros();
  lateSamples = 0;
  for (unsigned i = 0; i < count; ++i) {
    while ((int32_t)(micros() - next) < 0) delayMicroseconds(10);
    if ((int32_t)(micros() - next) >= (int32_t)SAMPLE_US) ++lateSamples;
    samples[i] = (uint16_t)analogReadMilliVolts(pin);
    next += SAMPLE_US;
  }
}
float meanSamples(unsigned count) {
  double sum = 0;
  for (unsigned i = 0; i < count; ++i) sum += samples[i];
  return sum / count;
}
float measureMean(uint8_t pin) { capture(pin, BASE_SAMPLES); return meanSamples(BASE_SAMPLES); }
void printResult(const Result &r) {
  Serial.printf("%s: %s | %s", r.name, r.done ? (r.pass ? "PASS" : "FAIL") : "NOT RUN", r.values.c_str());
  if (r.done && !r.pass) Serial.printf(" | %s", r.hint.c_str());
  Serial.println();
}
void biasTest() {
  Serial.println("Bias: settling for 3 s; leave sensor untouched."); delay(3000);
  float v = measureMean(SIGNAL_PIN);
  bias.done = true; bias.pass = v >= BIAS_MIN && v <= BIAS_MAX && lateSamples == 0;
  bias.values = String(v, 1) + " mV; late=" + String(lateSamples);
  if (lateSamples) bias.hint = "Sampling deadline missed; repeat test.";
  else if (v >= HIGH_MIN && v <= HIGH_MAX) bias.hint = "bias too high: half B / pin 5 problem (MCP602 input ceiling about 2.1 V); verify with meter";
  else if (v < LOW_CHECK) bias.hint = "check R4, the jumper to GPIO34, op-amp power, pin 1";
  else if (v > HIGH_CHECK) bias.hint = "check pin 1 - pin 2 link, pin 4 ground";
  else bias.hint = "Bias outside range; check divider, reference and connections with meter.";
  printResult(bias);
}
void opampTest() {
#if ENABLE_PIN5_ADC && ENABLE_PIN7_ADC && REF_MODE == NORMAL
  Serial.println("Op-amp check: keep sensor still during sequential measurements.");
  float p5 = measureMean(PIN5_ADC); unsigned late = lateSamples;
  float p7 = measureMean(PIN7_ADC); late += lateSamples;
  float out = measureMean(SIGNAL_PIN); late += lateSamples;
  bool a = p5 >= BIAS_MIN && p5 <= BIAS_MAX;
  bool b = fabsf(p7 - p5) <= MATCH_MV;
  bool c = fabsf(out - p7) <= MATCH_MV;
  Serial.printf("Pin 5 %.1f mV: %s%s\n", p5, a ? "PASS" : "FAIL", a ? "" : " - half B input leaking / bias node pulled; verify with meter");
  Serial.printf("Pin 7 %.1f mV vs pin 5: %s\n", p7, b ? "PASS" : "FAIL");
  Serial.printf("GPIO34 %.1f mV vs pin 7: %s\n", out, c ? "PASS" : "FAIL");
  opamp.done = true; opamp.pass = a && b && c && late == 0;
  opamp.values = "pin5=" + String(p5, 1) + ", pin7=" + String(p7, 1) + ", pin1=" + String(out, 1) + " mV; late=" + String(late);
  opamp.hint = "Check bias node and follower connections with a meter; ADC error may exceed 30 mV.";
  printResult(opamp);
#else
  opamp.done = false; opamp.values = "Requires both optional wires enabled in NORMAL mode";
  Serial.println("Op-amp check NOT RUN: enable GPIO33 and GPIO35 wires in NORMAL mode.");
#endif
}
void noiseTest() {
  Serial.println("Noise: don't touch anything for 5 s.");
  capture(SIGNAL_PIN, NOISE_SAMPLES);
  float avg = meanSamples(NOISE_SAMPLES); double sumsq = 0;
  int lo = 65535, hi = 0;
  for (unsigned i = 0; i < NOISE_SAMPLES; ++i) {
    float d = samples[i] - avg; sumsq += d * d;
    if (samples[i] < lo) lo = samples[i]; if (samples[i] > hi) hi = samples[i];
  }
  noiseRms = sqrt(sumsq / NOISE_SAMPLES); haveNoise = lateSamples == 0;
  noise.done = true; noise.pass = haveNoise && noiseRms < NOISE_PASS_MV;
  noise.values = "mean=" + String(avg, 1) + " mV, RMS=" + String(noiseRms, 2) + " mV, p-p=" + String(hi-lo) + " mV; late=" + String(lateSamples);
  noise.hint = lateSamples ? "Sampling deadline missed; repeat test." : "Check wiring, grounding, power and mechanical vibration.";
  printResult(noise);
}
void tapTest() {
  if (!haveNoise) { Serial.println("A valid noise measurement is needed first."); noiseTest(); }
  if (!haveNoise) { tap.done = false; tap.values = "No valid noise measurement"; return; }
  Serial.println("Tap: baseline measurement; keep still for 1 s.");
  float baseline = measureMean(SIGNAL_PIN); unsigned late = lateSamples;
  float threshold = fmaxf(TAP_NOISE_MULT * noiseRms, TAP_MIN_MV);
  // Require 50 consecutive in-band samples so a zero crossing is not settling.
  float settleBand = fmaxf(SETTLE_NOISE_MULT * noiseRms, 1.0f);
  Serial.printf("Tap the sensor 5 times in 10 s (threshold %.1f mV).\n", threshold);
  capture(SIGNAL_PIN, TAP_SAMPLES); late += lateSamples;
  struct Event { unsigned start, settle; float up, down; bool settled; } events[MAX_EVENTS];
  unsigned count = 0; int current = -1; unsigned quiet = 0, lastStart = 0; bool clipped = false;
  for (unsigned i = 0; i < TAP_SAMPLES; ++i) {
    float d = samples[i] - baseline;
    if (samples[i] >= CLIP_HIGH || samples[i] <= CLIP_LOW) clipped = true;
    if (fabsf(d) > threshold && (count == 0 || i - lastStart >= LOCKOUT_MS)) {
      if (count < MAX_EVENTS) {
        current = (int)count; events[count] = {i, 0, 0, 0, false}; ++count; lastStart = i; quiet = 0;
      }
    }
    if (current >= 0) {
      Event &e = events[current];
      if (d > e.up) e.up = d; if (d < e.down) e.down = d;
      if (fabsf(d) <= settleBand) ++quiet; else quiet = 0;
      if (quiet >= SETTLE_SAMPLES) { e.settled = true; e.settle = i + 1 - SETTLE_SAMPLES - e.start; current = -1; quiet = 0; }
    }
  }
  bool settled = true;
  for (unsigned i = 0; i < count; ++i) {
    Serial.printf("Tap %u: peak +%.1f / %.1f mV, ", i+1, events[i].up, events[i].down);
    if (events[i].settled) Serial.printf("settled %u ms\n", events[i].settle);
    else Serial.println("did not settle before next event/end");
    if (!events[i].settled || events[i].settle >= MAX_SETTLE_MS) settled = false;
  }
  Serial.println("Keep still for 1 s for the post-tap baseline.");
  float after = measureMean(SIGNAL_PIN); late += lateSamples;
  tap.done = true; tap.pass = count >= MIN_TAPS && count <= MAX_TAPS && settled && fabsf(after-baseline) <= BASE_DRIFT_MV && !clipped && late == 0;
  tap.values = "events=" + String(count) + ", baseline=" + String(baseline, 1) + "/" + String(after, 1) + " mV, clipping=" + String(clipped ? "yes" : "no") + ", late=" + String(late);
  tap.hint = count == 0 ? "no signal: check sensor power, S wire, C2, R3" : "Check tap timing, baseline drift, settling and clipping; repeat with a meter/scope.";
  if (clipped) Serial.println("CLIPPING WARNING: sample reached screening limit; this alone does not prove actual clipping.");
  printResult(tap);
}
void help() { Serial.println("a=all, b=bias, o=op-amp, n=noise, t=tap, s=plotter stream, h=help"); }
void stream() {
  Serial.println("Keep still for 1 s to measure baseline, then open Serial Plotter; any key stops.");
  float baseline = measureMean(SIGNAL_PIN); uint32_t next = micros();
  while (!Serial.available()) {
    while ((int32_t)(micros()-next) < 0) delayMicroseconds(50);
    Serial.printf("signal:%lu,baseline:%.1f\n", (unsigned long)analogReadMilliVolts(SIGNAL_PIN), baseline);
    next += 5000;
    if ((int32_t)(micros()-next) > 5000) next = micros();
  }
  while (Serial.available()) Serial.read();
  Serial.println("Stream stopped."); help();
}
void setup() {
  Serial.begin(115200); analogReadResolution(12);
  analogSetPinAttenuation(SIGNAL_PIN, ADC_11db);
#if ENABLE_PIN5_ADC
  analogSetPinAttenuation(PIN5_ADC, ADC_11db);
#endif
#if ENABLE_PIN7_ADC
  analogSetPinAttenuation(PIN7_ADC, ADC_11db);
#endif
  Serial.println("\nPiezo unity-gain buffer test. ADC readings are approximate; verify DC voltages with a meter.");
  Serial.println(REF_MODE == NORMAL ? "Reference: NORMAL" : "Reference: WORKAROUND"); help();
}
void loop() {
  if (!Serial.available()) { delay(10); return; }
  char c = (char)Serial.read();
  switch (c) {
    case 'a': biasTest(); opampTest(); noiseTest(); tapTest(); Serial.println("\nSummary:"); printResult(bias); printResult(opamp); printResult(noise); printResult(tap); break;
    case 'b': biasTest(); break; case 'o': opampTest(); break; case 'n': noiseTest(); break;
    case 't': tapTest(); break; case 's': while (Serial.available()) Serial.read(); stream(); break;
    case 'h': help(); break; default: break;
  }
}
