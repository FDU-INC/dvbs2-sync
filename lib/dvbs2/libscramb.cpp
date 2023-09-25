#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const int SEQ_LEN = (1 << 18) - 1;

extern "C" int8_t *malloc_scramble_seq(int scram_idx) {
  const int n = 10949 * scram_idx;

  int8_t *x = (int8_t *)malloc(SEQ_LEN * sizeof(int8_t));
  int8_t *y = (int8_t *)malloc(SEQ_LEN * sizeof(int8_t));
  for (int i = 0; i < 18; i++) {
    x[i] = 0;
    y[i] = 1;
  }
  x[0] = 1;

  for (int i = 18; i < SEQ_LEN; i++) {
    x[i] = (x[i - 11] + x[i - 18]) % 2;
    y[i] = (y[i - 8] + y[i - 11] + y[i - 13] + y[i - 18]) % 2;
  }

  int8_t *zn = (int8_t *)malloc(SEQ_LEN * sizeof(int8_t));
  for (int i = 0; i < SEQ_LEN; i++) {
    zn[i] = (x[(i + n) % SEQ_LEN] + y[i]) % 2;
  }
  free(x);
  free(y);

  int8_t *rn = (int8_t *)malloc(SEQ_LEN * sizeof(int8_t));
  for (int i = 0; i < SEQ_LEN; i++) {
    rn[i] = 2 * zn[(i + 131072) % SEQ_LEN] + zn[i];
  }
  free(zn);

  return rn;
}

extern "C" void free_scramble_seq(int8_t *handle) { free(handle); }

extern "C" int get_seq_len() { return SEQ_LEN; }
