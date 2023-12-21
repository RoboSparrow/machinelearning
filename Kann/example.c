/**
 * A complete example
 *
 * This example learns to count the number of "1" bits in an integer (i.e. popcount):
 */

// to compile and run: gcc -O2 example.c kann.c kautodiff.c -lm && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include "kann/kann.h"

int main(void)
{
    int i, k, max_bit = 20, n_samples = 30000, mask = (1<<max_bit)-1, n_err, max_k;
    float **x, **y, max, *x1;
    kad_node_t *t;
    kann_t *ann;

    // construct an MLP with one hidden layers

    t = kann_layer_input(max_bit);
    t = kad_relu(kann_layer_dense(t, 64));
    t = kann_layer_cost(t, max_bit + 1, KANN_C_CEM); // output uses 1-hot encoding
    ann = kann_new(t, 0);

    // generate training data

    x = (float**)calloc(n_samples, sizeof(float*));
    y = (float**)calloc(n_samples, sizeof(float*));

    for (i = 0; i < n_samples; ++i) {
        int c, a = kad_rand(0) & (mask>>1);
        x[i] = (float*)calloc(max_bit, sizeof(float));
        y[i] = (float*)calloc(max_bit + 1, sizeof(float));

        for (k = c = 0; k < max_bit; ++k) {
            x[i][k] = (float)(a>>k&1), c += (a>>k&1);
        }
        y[i][c] = 1.0f; // c is ranged from 0 to max_bit inclusive
    }

    // train

    kann_train_fnn1(ann, 0.001f, 64, 50, 10, 0.1f, n_samples, x, y);

    // predict

    x1 = (float*)calloc(max_bit, sizeof(float));
    for (i = n_err = 0; i < n_samples; ++i) {
        int c, a = kad_rand(0) & (mask>>1); // generating a new number
        const float *y1;
        for (k = c = 0; k < max_bit; ++k) {
            x1[k] = (float)(a>>k&1), c += (a>>k&1);
        }

        y1 = kann_apply1(ann, x1);
        for (k = 0, max_k = -1, max = -1.0f; k <= max_bit; ++k) { // find the max
            if (max < y1[k]) {
                max = y1[k];
                max_k = k;
            }
        }
        if (max_k != c) {
            ++n_err;
        }
    }

    fprintf(stderr, "Test error rate: %.2f%%\n", 100.0 * n_err / n_samples);
    kann_delete(ann); // TODO: also to free x, y and x1

    return 0;
}
