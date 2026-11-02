/* Host parity test: stream fixture windows through logmel_push in uneven chunks and compare with
 * the Python training features. Usage: parity_test <fixtures.bin>. Exit code 0 when every int8
 * value is within 1 LSB. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../logmel.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s fixtures.bin\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    char magic[4];
    int32_t count, zp;
    float scale;
    if (!f || fread(magic, 1, 4, f) != 4 || memcmp(magic, "LMEL", 4) || fread(&count, 4, 1, f) != 1 ||
        fread(&scale, 4, 1, f) != 1 || fread(&zp, 4, 1, f) != 1) {
        fprintf(stderr, "cannot read fixture header\n");
        return 2;
    }
    static int16_t pcm[LOGMEL_WINDOW_SAMPLES];
    static float expect_f[LOGMEL_OUTPUT_COUNT], got_f[LOGMEL_OUTPUT_COUNT];
    static int8_t expect_q[LOGMEL_OUTPUT_COUNT], got_q[LOGMEL_OUTPUT_COUNT];
    static logmel_stream_t s;
    logmel_init(&s, scale, zp, got_q, got_f);
    static const size_t chunks[] = {1, 127, 333, 500, 1000, 64, 4096};  /* exercise the streaming path */
    double max_abs = 0, sum_abs = 0;
    long differ = 0, over_one = 0, total = 0;
    for (int w = 0; w < count; ++w) {
        if (fread(pcm, 2, LOGMEL_WINDOW_SAMPLES, f) != LOGMEL_WINDOW_SAMPLES ||
            fread(expect_f, 4, LOGMEL_OUTPUT_COUNT, f) != LOGMEL_OUTPUT_COUNT ||
            fread(expect_q, 1, LOGMEL_OUTPUT_COUNT, f) != LOGMEL_OUTPUT_COUNT) {
            fprintf(stderr, "truncated fixture at window %d\n", w);
            return 2;
        }
        size_t done = 0, c = (size_t)w;
        int ready = 0;
        while (!ready) {
            size_t want = chunks[c++ % (sizeof chunks / sizeof chunks[0])];
            if (want > LOGMEL_WINDOW_SAMPLES - done) want = LOGMEL_WINDOW_SAMPLES - done;
            done += logmel_push(&s, pcm + done, want, &ready);
        }
        if (done != LOGMEL_WINDOW_SAMPLES) {
            fprintf(stderr, "window %d finished after %zu samples\n", w, done);
            return 1;
        }
        double wmax = 0;
        long wdiff = 0;
        for (int i = 0; i < LOGMEL_OUTPUT_COUNT; ++i) {
            const double e = fabs((double)got_f[i] - (double)expect_f[i]);
            const int dq = abs((int)got_q[i] - (int)expect_q[i]);
            wmax = e > wmax ? e : wmax;
            sum_abs += e;
            wdiff += dq != 0;
            over_one += dq > 1;
        }
        max_abs = wmax > max_abs ? wmax : max_abs;
        differ += wdiff;
        total += LOGMEL_OUTPUT_COUNT;
        printf("window %2d: max |float err| %.3e, int8 values differing %ld/%d\n", w, wmax, wdiff,
               LOGMEL_OUTPUT_COUNT);
    }
    fclose(f);
    printf("SUMMARY windows=%d values=%ld max_abs_float_err=%.3e mean_abs_float_err=%.3e "
           "int8_differ=%ld (%.4f%%) int8_over_1lsb=%ld\n", count, total, max_abs, sum_abs / total, differ,
           100.0 * differ / total, over_one);
    printf("STATE bytes sizeof(logmel_stream_t)=%zu (ring %zu, fft scratch %zu, power %zu)\n",
           sizeof(logmel_stream_t), sizeof s.ring, sizeof s.re + sizeof s.im, sizeof s.power);
    return over_one == 0 ? 0 : 1;
}
