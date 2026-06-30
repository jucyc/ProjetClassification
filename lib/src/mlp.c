#include "mlp.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double sigmoid_deriv(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

static double uniform() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

MLP* mlp_create(int input_size, int hidden_size, int output_size) {
    MLP* net = (MLP*)malloc(sizeof(MLP));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    
    double scale_in = sqrt(2.0 / input_size);
    double scale_hid = sqrt(2.0 / hidden_size);
    
    net->W1 = (double*)malloc(input_size * hidden_size * sizeof(double));
    net->b1 = (double*)calloc(hidden_size, sizeof(double));
    net->W2 = (double*)malloc(hidden_size * output_size * sizeof(double));
    net->b2 = (double*)calloc(output_size, sizeof(double));
    
    if (!net->W1 || !net->b1 || !net->W2 || !net->b2) {
        mlp_destroy(net);
        return NULL;
    }
    
    for (int i = 0; i < input_size * hidden_size; i++)
        net->W1[i] = uniform() * scale_in;
    for (int i = 0; i < hidden_size * output_size; i++)
        net->W2[i] = uniform() * scale_hid;
    
    return net;
}

void mlp_destroy(MLP* net) {
    if (net) {
        if (net->W1) free(net->W1);
        if (net->b1) free(net->b1);
        if (net->W2) free(net->W2);
        if (net->b2) free(net->b2);
        free(net);
    }
}

static void forward(MLP* net, double* x, double* h, double* out) {
    for (int j = 0; j < net->hidden_size; j++) {
        double z = net->b1[j];
        for (int i = 0; i < net->input_size; i++) {
            z += net->W1[i * net->hidden_size + j] * x[i];
        }
        h[j] = sigmoid(z);
    }
    
    double max_val = -1e18;
    for (int k = 0; k < net->output_size; k++) {
        double z = net->b2[k];
        for (int j = 0; j < net->hidden_size; j++) {
            z += net->W2[j * net->output_size + k] * h[j];
        }
        out[k] = z;
        if (z > max_val) max_val = z;
    }
    
    double sum = 0;
    for (int k = 0; k < net->output_size; k++) {
        out[k] = exp(out[k] - max_val);
        sum += out[k];
    }
    for (int k = 0; k < net->output_size; k++) {
        out[k] /= sum;
    }
}

int mlp_predict(MLP* net, double* x) {
    double* h = (double*)malloc(net->hidden_size * sizeof(double));
    double* out = (double*)malloc(net->output_size * sizeof(double));
    if (!h || !out) return 0;
    
    forward(net, x, h, out);
    
    int pred = 0;
    for (int k = 1; k < net->output_size; k++) {
        if (out[k] > out[pred]) pred = k;
    }
    
    free(h);
    free(out);
    return pred;
}

double mlp_evaluate(MLP* net, double* X, int* y, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        double* x = X + (long)i * net->input_size;
        if (mlp_predict(net, x) == y[i]) {
            correct++;
        }
    }
    return 100.0 * correct / n_samples;
}

void mlp_train(MLP* net, double* X, int* y, int n_samples,
               double lr, int epochs, int batch_size) {
    double *h = (double*)malloc(net->hidden_size * sizeof(double));
    double *out = (double*)malloc(net->output_size * sizeof(double));
    double *grad_W1 = (double*)calloc(net->input_size * net->hidden_size, sizeof(double));
    double *grad_b1 = (double*)calloc(net->hidden_size, sizeof(double));
    double *grad_W2 = (double*)calloc(net->hidden_size * net->output_size, sizeof(double));
    double *grad_b2 = (double*)calloc(net->output_size, sizeof(double));
    
    if (!h || !out || !grad_W1 || !grad_b1 || !grad_W2 || !grad_b2) {
        printf("Memory allocation error\n");
        return;
    }
    
    srand((unsigned int)time(NULL));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        memset(grad_W1, 0, net->input_size * net->hidden_size * sizeof(double));
        memset(grad_b1, 0, net->hidden_size * sizeof(double));
        memset(grad_W2, 0, net->hidden_size * net->output_size * sizeof(double));
        memset(grad_b2, 0, net->output_size * sizeof(double));
        
        int* idx = (int*)malloc(n_samples * sizeof(int));
        for (int i = 0; i < n_samples; i++) idx[i] = i;
        for (int i = n_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = idx[i];
            idx[i] = idx[j];
            idx[j] = tmp;
        }
        
        for (int b = 0; b < n_samples; b++) {
            int sample_idx = idx[b];
            double* x = X + (long)sample_idx * net->input_size;
            
            forward(net, x, h, out);
            
            double* delta_out = (double*)malloc(net->output_size * sizeof(double));
            for (int k = 0; k < net->output_size; k++) {
                delta_out[k] = out[k] - (k == y[sample_idx] ? 1.0 : 0.0);
            }
            
            for (int j = 0; j < net->hidden_size; j++) {
                for (int k = 0; k < net->output_size; k++) {
                    grad_W2[j * net->output_size + k] += delta_out[k] * h[j];
                }
            }
            for (int k = 0; k < net->output_size; k++) {
                grad_b2[k] += delta_out[k];
            }
            
            double* delta_hid = (double*)calloc(net->hidden_size, sizeof(double));
            for (int j = 0; j < net->hidden_size; j++) {
                for (int k = 0; k < net->output_size; k++) {
                    delta_hid[j] += delta_out[k] * net->W2[j * net->output_size + k];
                }
                delta_hid[j] *= sigmoid_deriv(h[j]);
            }
            
            for (int f = 0; f < net->input_size; f++) {
                for (int j = 0; j < net->hidden_size; j++) {
                    grad_W1[f * net->hidden_size + j] += delta_hid[j] * x[f];
                }
            }
            for (int j = 0; j < net->hidden_size; j++) {
                grad_b1[j] += delta_hid[j];
            }
            
            free(delta_out);
            free(delta_hid);
        }
        free(idx);
        
        double inv_batch = 1.0 / batch_size;
        for (int i = 0; i < net->input_size * net->hidden_size; i++) {
            net->W1[i] -= lr * grad_W1[i] * inv_batch;
        }
        for (int i = 0; i < net->hidden_size; i++) {
            net->b1[i] -= lr * grad_b1[i] * inv_batch;
        }
        for (int i = 0; i < net->hidden_size * net->output_size; i++) {
            net->W2[i] -= lr * grad_W2[i] * inv_batch;
        }
        for (int i = 0; i < net->output_size; i++) {
            net->b2[i] -= lr * grad_b2[i] * inv_batch;
        }
        
        if (epoch % 100 == 0 || epoch == epochs - 1) {
            double acc = mlp_evaluate(net, X, y, n_samples);
            printf("Epoch %d - Accuracy: %.2f%%\n", epoch, acc);
        }
    }
    
    free(h);
    free(out);
    free(grad_W1);
    free(grad_b1);
    free(grad_W2);
    free(grad_b2);
}

void mlp_save(MLP* net, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("Error opening %s\n", path);
        return;
    }
    fwrite(&net->input_size, sizeof(int), 1, f);
    fwrite(&net->hidden_size, sizeof(int), 1, f);
    fwrite(&net->output_size, sizeof(int), 1, f);
    fwrite(net->W1, sizeof(double), net->input_size * net->hidden_size, f);
    fwrite(net->b1, sizeof(double), net->hidden_size, f);
    fwrite(net->W2, sizeof(double), net->hidden_size * net->output_size, f);
    fwrite(net->b2, sizeof(double), net->output_size, f);
    fclose(f);
    printf("Model saved: %s\n", path);
}

MLP* mlp_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Error opening %s\n", path);
        return NULL;
    }
    int input_size, hidden_size, output_size;
    fread(&input_size, sizeof(int), 1, f);
    fread(&hidden_size, sizeof(int), 1, f);
    fread(&output_size, sizeof(int), 1, f);
    
    MLP* net = mlp_create(input_size, hidden_size, output_size);
    if (!net) {
        fclose(f);
        return NULL;
    }
    
    fread(net->W1, sizeof(double), input_size * hidden_size, f);
    fread(net->b1, sizeof(double), hidden_size, f);
    fread(net->W2, sizeof(double), hidden_size * output_size, f);
    fread(net->b2, sizeof(double), output_size, f);
    fclose(f);
    printf("Model loaded: %s\n", path);
    return net;
}