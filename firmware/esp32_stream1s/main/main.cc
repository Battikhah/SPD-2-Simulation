/**
 * ESP32-WROOM32 example: streaming log-mel front end (firmware/logmel) feeding the stream1s_w8 int8
 * model under TFLite Micro. Derived from the reference benchmark harness
 * (ML-for-Senior deployment/esp32/main/main.cc). Not yet built or run on a board.
 *
 * Audio arrives in arbitrary chunks (here: a real 1 s fixture replayed ten times, in 160-sample
 * chunks as an I2S driver might deliver). The front end writes int8 log-mel directly into the TFLM
 * input tensor; each completed 1 s window is classified, and ten window probabilities are averaged
 * into one 10 s decision. No 1 s or 10 s audio buffer exists.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "logmel.h"
#include "model_data.h"
#include "fixture.h"

#ifndef TREEVIBES_ARENA_BYTES
#define TREEVIBES_ARENA_BYTES (64 * 1024)  // host estimate ~38 KiB; measure arena_used_bytes on the board
#endif

static logmel_stream_t g_logmel;  // 6,184 B static state

extern "C" void app_main() {
    constexpr char tag[] = "stream1s";
    const auto* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) { ESP_LOGE(tag, "Model schema mismatch"); return; }
    auto* arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, TREEVIBES_ARENA_BYTES,
                                                                 MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!arena) { ESP_LOGE(tag, "Arena allocation failed"); return; }
    // Operators emitted by esp32_export.py for stream1s_w8 (see device_report.json).
    auto* resolver = new tflite::MicroMutableOpResolver<8>();
    resolver->AddAdd(); resolver->AddConv2D(); resolver->AddFullyConnected(); resolver->AddLogistic();
    resolver->AddMaxPool2D(); resolver->AddMean(); resolver->AddMul();
    auto* interpreter = new tflite::MicroInterpreter(model, *resolver, arena, TREEVIBES_ARENA_BYTES);
    if (interpreter->AllocateTensors() != kTfLiteOk) { ESP_LOGE(tag, "Tensor allocation failed"); return; }
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);
    if (input->type != kTfLiteInt8 || input->bytes != LOGMEL_OUTPUT_COUNT) { ESP_LOGE(tag, "Bad input"); return; }
    logmel_init(&g_logmel, input->params.scale, input->params.zero_point, input->data.int8, nullptr);

    constexpr int kWindows = 10, kChunk = 160;
    float prob_sum = 0;
    int64_t frontend_us = 0, invoke_us = 0;
    int first_output = 0, input_sum = 0;
    for (int w = 0; w < kWindows; ++w) {
        int ready = 0;
        size_t done = 0;
        const int64_t t0 = esp_timer_get_time();
        while (!ready) {
            const size_t n = std::min<size_t>(kChunk, LOGMEL_WINDOW_SAMPLES - done);
            done += logmel_push(&g_logmel, kFixturePcm + done, n, &ready);
        }
        const int64_t t1 = esp_timer_get_time();
        if (w == 0) for (int i = 0; i < LOGMEL_OUTPUT_COUNT; ++i) input_sum += input->data.int8[i];
        if (interpreter->Invoke() != kTfLiteOk) { ESP_LOGE(tag, "Invoke failed"); return; }
        const int64_t t2 = esp_timer_get_time();
        frontend_us += t1 - t0;
        invoke_us += t2 - t1;
        const int q = output->data.int8[0];
        if (w == 0) first_output = q;
        prob_sum += (q - output->params.zero_point) * output->params.scale;  // LOGISTIC output: probability
    }
    // The decision threshold must come from a final model/threshold frozen outside these folds.
    printf("{\"model_sha256\":\"%s\",\"scope\":\"logmel+inference example\",\"windows\":%d,"
           "\"mean_probability\":%.4f,\"input_sum_pass\":%s,\"fixture_output_delta\":%d,"
           "\"frontend_us_per_window\":%lld,\"invoke_us_per_window\":%lld,\"arena_used_bytes\":%u,"
           "\"arena_reserved_bytes\":%u,\"free_internal_heap\":%u,\"largest_free_block\":%u}\n",
           kModelHash, kWindows, prob_sum / kWindows, input_sum == kExpectedInputSum ? "true" : "false",
           std::abs(first_output - kExpectedOutput), static_cast<long long>(frontend_us / kWindows),
           static_cast<long long>(invoke_us / kWindows), static_cast<unsigned>(interpreter->arena_used_bytes()),
           TREEVIBES_ARENA_BYTES,
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
           static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
}
