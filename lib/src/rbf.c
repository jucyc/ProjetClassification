#include "rbf.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

static double uniform() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

static double squared_distance(double* a, double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

RBF* rbf_create(int input_size, int n_centers, int output_size, double gamma) {
    RBF* net = (RBF*)malloc(sizeof(RBF));
    if (!net) return NULL;

    net->input_size = input_size;
    net->n_centers = n_centers;
    net->output_size = output_size;
    net->gamma = gamma;

    net->centers = (double*)calloc((long)n_centers * input_size, sizeof(double));
    net->W = (double*)malloc((long)n_centers * output_size * sizeof(double));
    net->b = (double*)calloc(output_size, sizeof(double));

    if (!net->centers || !net->W || !net->b) {
        rbf_destroy(net);
        return NULL;
    }

    double scale = sqrt(2.0 / n_centers);
    for (long i = 0; i < (long)n_centers * output_size; i++) {
        net->W[i] = uniform() * scale;
    }

    return net;
}

void rbf_destroy(RBF* net) {
    if (net) {
        if (net->centers) free(net->centers);
        if (net->W) free(net->W);
        if (net->b) free(net->b);
        free(net);
    }
}

void rbf_fit_centers(RBF* net, double* X, int n_samples, int kmeans_iters) {
    int n = net->input_size;
    int k = net->n_centers;

    srand((unsigned int)time(NULL));
    int* chosen = (int*)malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) chosen[i] = i;
    for (int i = n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = chosen[i]; chosen[i] = chosen[j]; chosen[j] = tmp;
    }
    for (int c = 0; c < k; c++) {
        double* src = X + (long)chosen[c % n_samples] * n;
        memcpy(net->centers + (long)c * n, src, n * sizeof(double));
    }
    free(chosen);

    int* assignment = (int*)malloc(n_samples * sizeof(int));
    double* sums = (double*)malloc((long)k * n * sizeof(double));
    int* counts = (int*)malloc(k * sizeof(int));

    for (int iter = 0; iter < kmeans_iters; iter++) {
        for (int i = 0; i < n_samples; i++) {
            double* x = X + (long)i * n;
            int best = 0;
            double best_dist = squared_distance(x, net->centers, n);
            for (int c = 1; c < k; c++) {
                double dist = squared_distance(x, net->centers + (long)c * n, n);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            assignment[i] = best;
        }

        memset(sums, 0, (long)k * n * sizeof(double));
        memset(counts, 0, k * sizeof(int));

        for (int i = 0; i < n_samples; i++) {
            int c = assignment[i];
            double* x = X + (long)i * n;
            for (int f = 0; f < n; f++) {
                sums[(long)c * n + f] += x[f];
            }
            counts[c]++;
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int f = 0; f < n; f++) {
                    net->centers[(long)c * n + f] = sums[(long)c * n + f] / counts[c];
                }
            }
        }
    }

    free(assignment);
    free(sums);
    free(counts);

    printf("K-means termine (%d iterations, %d centres)\n", kmeans_iters, k);
}

static void forward(RBF* net, double* x, double* phi, double* out) {
    int n = net->input_size;
    int k = net->n_centers;

    for (int c = 0; c < k; c++) {
        double dist2 = squared_distance(x, net->centers + (long)c * n, n);
        phi[c] = exp(-net->gamma * dist2);
    }

    double max_val = -1e18;
    for (int o = 0; o < net->output_size; o++) {
        double z = net->b[o];
        for (int c = 0; c < k; c++) {
            z += net->W[(long)c * net->output_size + o] * phi[c];
        }
        out[o] = z;
        if (z > max_val) max_val = z;
    }

    double sum = 0.0;
    for (int o = 0; o < net->output_size; o++) {
        out[o] = exp(out[o] - max_val);
        sum += out[o];
    }
    for (int o = 0; o < net->output_size; o++) {
        out[o] /= sum;
    }
}

int rbf_predict(RBF* net, double* x) {
    double* phi = (double*)malloc(net->n_centers * sizeof(double));
    double* out = (double*)malloc(net->output_size * sizeof(double));
    if (!phi || !out) return 0;

    forward(net, x, phi, out);

    int pred = 0;
    for (int o = 1; o < net->output_size; o++) {
        if (out[o] > out[pred]) pred = o;
    }

    free(phi);
    free(out);
    return pred;
}

double rbf_evaluate(RBF* net, double* X, int* y, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        double* x = X + (long)i * net->input_size;
        if (rbf_predict(net, x) == y[i]) {
            correct++;
        }
    }
    return 100.0 * correct / n_samples;
}

void rbf_train(RBF* net, double* X, int* y, int n_samples,
               double lr, int epochs, int batch_size) {
    int k = net->n_centers;
    int n_out = net->output_size;

    double* phi = (double*)malloc(k * sizeof(double));
    double* out = (double*)malloc(n_out * sizeof(double));
    double* grad_W = (double*)calloc((long)k * n_out, sizeof(double));
    double* grad_b = (double*)calloc(n_out, sizeof(double));

    if (!phi || !out || !grad_W || !grad_b) {
        printf("Memory allocation error\n");
        return;
    }

    srand((unsigned int)time(NULL));

    for (int epoch = 0; epoch < epochs; epoch++) {
        memset(grad_W, 0, (long)k * n_out * sizeof(double));
        memset(grad_b, 0, n_out * sizeof(double));

        int* idx = (int*)malloc(n_samples * sizeof(int));
        for (int i = 0; i < n_samples; i++) idx[i] = i;
        for (int i = n_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }

        for (int b_i = 0; b_i < n_samples; b_i++) {
            int sample_idx = idx[b_i];
            double* x = X + (long)sample_idx * net->input_size;

            forward(net, x, phi, out);

            double* delta_out = (double*)malloc(n_out * sizeof(double));
            for (int o = 0; o < n_out; o++) {
                delta_out[o] = out[o] - (o == y[sample_idx] ? 1.0 : 0.0);
            }

            for (int c = 0; c < k; c++) {
                for (int o = 0; o < n_out; o++) {
                    grad_W[(long)c * n_out + o] += delta_out[o] * phi[c];
                }
            }
            for (int o = 0; o < n_out; o++) {
                grad_b[o] += delta_out[o];
            }

            free(delta_out);
        }
        free(idx);

        double inv_batch = 1.0 / batch_size;
        for (long i = 0; i < (long)k * n_out; i++) {
            net->W[i] -= lr * grad_W[i] * inv_batch;
        }
        for (int o = 0; o < n_out; o++) {
            net->b[o] -= lr * grad_b[o] * inv_batch;
        }

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            double acc = rbf_evaluate(net, X, y, n_samples);
            printf("Epoch %d - Accuracy: %.2f%%\n", epoch, acc);
        }
    }

    free(phi);
    free(out);
    free(grad_W);
    free(grad_b);
}

void rbf_save(RBF* net, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("Error opening %s\n", path);
        return;
    }
    fwrite(&net->input_size, sizeof(int), 1, f);
    fwrite(&net->n_centers, sizeof(int), 1, f);
    fwrite(&net->output_size, sizeof(int), 1, f);
    fwrite(&net->gamma, sizeof(double), 1, f);
    fwrite(net->centers, sizeof(double), (long)net->n_centers * net->input_size, f);
    fwrite(net->W, sizeof(double), (long)net->n_centers * net->output_size, f);
    fwrite(net->b, sizeof(double), net->output_size, f);
    fclose(f);
    printf("Model saved: %s\n", path);
}

RBF* rbf_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Error opening %s\n", path);
        return NULL;
    }
    int input_size, n_centers, output_size;
    double gamma;
    fread(&input_size, sizeof(int), 1, f);
    fread(&n_centers, sizeof(int), 1, f);
    fread(&output_size, sizeof(int), 1, f);
    fread(&gamma, sizeof(double), 1, f);

    RBF* net = rbf_create(input_size, n_centers, output_size, gamma);
    if (!net) {
        fclose(f);
        return NULL;
    }

    fread(net->centers, sizeof(double), (long)n_centers * input_size, f);
    fread(net->W, sizeof(double), (long)n_centers * output_size, f);
    fread(net->b, sizeof(double), output_size, f);
    fclose(f);
    printf("Model loaded: %s\n", path);
    return net;
}