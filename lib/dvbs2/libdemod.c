#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

inline double _square_dist(double xa, double ya, double xb, double yb) {
    return (xa - xb) * (xa - xb) + (ya - yb) * (ya - yb);
}

inline size_t _argmin(const double* x, size_t size) {
    if (size < 1) {
        fprintf(stderr, "0 element for min");
        exit(1);
    }
    double min_v = x[0];
    size_t id = 0;
    for (size_t i = 0; i < size; i++) {
        id = (x[i] < min_v) ? i : id;
        min_v = (x[i] < min_v) ? x[i] : min_v;
    }
    return id;
}

// for v2, in accordance with matlab code
void demod_qpsk(const double* real, const double* imag, size_t length, int8_t* out) {
    for (size_t i = 0; i < length; i++) {
        if (real[i] >= 0 && imag[i] >= 0) {
            out[2 * i] = 0;
            out[2 * i + 1] = 0;
        }
        if (real[i] > 0 && imag[i] < 0) {
            out[2 * i] = 1;
            out[2 * i + 1] = 0;
        }
        if (real[i] < 0 && imag[i] > 0) {
            out[2 * i] = 0;
            out[2 * i + 1] = 1;
        }
        if (real[i] < 0 && imag[i] < 0) {
            out[2 * i] = 1;
            out[2 * i + 1] = 1;
        }
    }
}

// in accordance with DVB-S2
void demod_8psk(const double* real, const double* imag, size_t length, int8_t* out) {
    double r = sqrt(real[0] * real[0] + imag[0] * imag[0]);
    double anchor_x = r * 0.707107;
    for (size_t i = 0; i < length; i++) {
        // first quadrant
        if (real[i] >= 0.0 && imag[i] >= 0.0) {
            out[3 * i + 1] = 0;
            double ds[3] = {
                _square_dist(real[i], imag[i], r, 0.0),
                _square_dist(real[i], imag[i], anchor_x, anchor_x),
                _square_dist(real[i], imag[i], 0.0, r),
            };
            size_t id = _argmin(ds, 3);

            if (id == 0) {
                // 001
                out[3 * i] = 0;
                out[3 * i + 2] = 1;
            } else if (id == 1) {
                // 000
                out[3 * i] = 0;
                out[3 * i + 2] = 0;
            } else if (id == 2) {
                // 100
                out[3 * i] = 1;
                out[3 * i + 2] = 0;
            } else {
                fprintf(stderr, "unreachable\n");
                exit(1);
            }
        }
        // forth quadrant
        if (real[i] > 0 && imag[i] < 0) {
            out[3 * i + 2] = 1;
            double ds[3] = {
                _square_dist(real[i], imag[i], 0.0, -r),
                _square_dist(real[i], imag[i], anchor_x, -anchor_x),
                _square_dist(real[i], imag[i], r, 0.0),
            };
            size_t id = _argmin(ds, 3);

            if (id == 0) {
                // 111
                out[3 * i] = 1;
                out[3 * i + 1] = 1;
            } else if (id == 1) {
                // 101
                out[3 * i] = 1;
                out[3 * i + 1] = 0;
            } else if (id == 2) {
                // 001
                out[3 * i] = 0;
                out[3 * i + 1] = 0;
            } else {
                fprintf(stderr, "unreachable\n");
                exit(1);
            }
        }
        // second quadrant
        if (real[i] < 0 && imag[i] > 0) {
            out[3 * i + 2] = 0;
            double ds[3] = {
                _square_dist(real[i], imag[i], 0.0, r),
                _square_dist(real[i], imag[i], -anchor_x, anchor_x),
                _square_dist(real[i], imag[i], -r, 0.0),
            };
            size_t id = _argmin(ds, 3);

            if (id == 0) {
                // 100
                out[3 * i] = 1;
                out[3 * i + 1] = 0;
            } else if (id == 1) {
                // 110
                out[3 * i] = 1;
                out[3 * i + 1] = 1;
            } else if (id == 2) {
                // 010
                out[3 * i] = 0;
                out[3 * i + 1] = 1;
            } else {
                fprintf(stderr, "unreachable\n");
                exit(1);
            }
        }
        // third quadrant
        if (real[i] < 0 && imag[i] < 0) {
            out[3 * i + 1] = 1;
            double ds[3] = {
                _square_dist(real[i], imag[i], -r, 0.0),
                _square_dist(real[i], imag[i], -anchor_x, -anchor_x),
                _square_dist(real[i], imag[i], 0.0, -r),
            };
            size_t id = _argmin(ds, 3);

            if (id == 0) {
                // 010
                out[3 * i] = 0;
                out[3 * i + 2] = 0;
            } else if (id == 1) {
                // 011
                out[3 * i] = 0;
                out[3 * i + 2] = 1;
            } else if (id == 2) {
                // 111
                out[3 * i] = 1;
                out[3 * i + 2] = 1;
            } else {
                fprintf(stderr, "unreachable\n");
                exit(1);
            }
        }
    }
}
