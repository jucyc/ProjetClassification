#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/mlp.h"

MLP* mlp_create(int* npl, int n_layers) {
    MLP* model = (MLP*)malloc(sizeof(MLP));
    model->n_layers = n_layers;
    model->L = n_layers - 1;
    model->is_trained = 0;

    model->d = (int*)malloc(n_layers * sizeof(int));
    for (int l = 0; l < n_layers; l++) {
        model->d[l] = npl[l];
    }

    model->W = (double***)malloc((model->L + 1) * sizeof(double**));
    model->W[0] = NULL;

    for (int l = 1; l <= model->L; l++) {
        int rows = model->d[l - 1] + 1; // +1 pour le biais de la couche l-1
        int cols = model->d[l] + 1;     // +1 pour l'indice 0 (non utilise) du neurone j

        model->W[l] = (double**)malloc(rows * sizeof(double*));
        for (int i = 0; i < rows; i++) {
            model->W[l][i] = (double*)malloc(cols * sizeof(double));
            for (int j = 0; j < cols; j++) {
                if (j == 0) {
                    model->W[l][i][j] = 0.0; // pas de poids vers un "biais" de sortie
                } else {
                    model->W[l][i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                }
            }
        }
    }

    model->X = (double**)malloc((model->L + 1) * sizeof(double*));
    model->deltas = (double**)malloc((model->L + 1) * sizeof(double*));

    for (int l = 0; l <= model->L; l++) {
        int size = model->d[l] + 1; // +1 pour le neurone de biais virtuel (indice 0)
        model->X[l] = (double*)malloc(size * sizeof(double));
        model->deltas[l] = (double*)malloc(size * sizeof(double));

        for (int j = 0; j < size; j++) {
            model->X[l][j] = (j == 0) ? 1.0 : 0.0;
            model->deltas[l][j] = 0.0;
        }
    }

    return model;
}

static void mlp_propagate(MLP* model, double* inputs) {
    for (int j = 1; j <= model->d[0]; j++) {
        model->X[0][j] = inputs[j - 1];
    }

    for (int l = 1; l <= model->L; l++) {
        for (int j = 1; j <= model->d[l]; j++) {
            double total = 0.0;
            for (int i = 0; i <= model->d[l - 1]; i++) {
                total += model->W[l][i][j] * model->X[l - 1][i];
            }
            model->X[l][j] = tanh(total);
        }
    }
}

double* mlp_predict_raw(MLP* model, double* x) {
    mlp_propagate(model, x);
    return &model->X[model->L][1];
}

int mlp_predict_class(MLP* model, double* x) {
    double* out = mlp_predict_raw(model, x);
    int best = 0;
    double best_val = out[0];
    for (int j = 1; j < model->d[model->L]; j++) {
        if (out[j] > best_val) {
            best_val = out[j];
            best = j;
        }
    }
    return best;
}

void mlp_train(MLP* model, double** X, double** Y, int n_samples,
               double learning_rate, int n_iterations) {
    int dL = model->d[model->L];

    for (int it = 0; it < n_iterations; it++) {
        int k = rand() % n_samples;
        double* xk = X[k];
        double* yk = Y[k];

        mlp_propagate(model, xk);

        for (int j = 1; j <= dL; j++) {
            double xj = model->X[model->L][j];
            model->deltas[model->L][j] = (1.0 - xj * xj) * (xj - yk[j - 1]);
        }

        for (int l = model->L; l >= 2; l--) {
            for (int i = 1; i <= model->d[l - 1]; i++) {
                double total = 0.0;
                for (int j = 1; j <= model->d[l]; j++) {
                    total += model->W[l][i][j] * model->deltas[l][j];
                }
                double xi = model->X[l - 1][i];
                model->deltas[l - 1][i] = (1.0 - xi * xi) * total;
            }
        }

        for (int l = 1; l <= model->L; l++) {
            for (int i = 0; i <= model->d[l - 1]; i++) {
                for (int j = 1; j <= model->d[l]; j++) {
                    model->W[l][i][j] -= learning_rate * model->X[l - 1][i] * model->deltas[l][j];
                }
            }
        }
    }

    model->is_trained = 1;
    printf("MLP entraine (%d iterations)\n", n_iterations);
}

void mlp_save(MLP* model, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    fwrite(&model->n_layers, sizeof(int), 1, f);
    fwrite(model->d, sizeof(int), model->n_layers, f);

    for (int l = 1; l <= model->L; l++) {
        int rows = model->d[l - 1] + 1;
        int cols = model->d[l] + 1;
        for (int i = 0; i < rows; i++) {
            fwrite(model->W[l][i], sizeof(double), cols, f);
        }
    }

    fclose(f);
}

MLP* mlp_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    int n_layers;
    fread(&n_layers, sizeof(int), 1, f);

    int* npl = (int*)malloc(n_layers * sizeof(int));
    fread(npl, sizeof(int), n_layers, f);

    MLP* model = mlp_create(npl, n_layers);
    free(npl);

    for (int l = 1; l <= model->L; l++) {
        int rows = model->d[l - 1] + 1;
        int cols = model->d[l] + 1;
        for (int i = 0; i < rows; i++) {
            fread(model->W[l][i], sizeof(double), cols, f);
        }
    }

    model->is_trained = 1;
    fclose(f);
    return model;
}

void mlp_free(MLP* model) {
    for (int l = 1; l <= model->L; l++) {
        int rows = model->d[l - 1] + 1;
        for (int i = 0; i < rows; i++) {
            free(model->W[l][i]);
        }
        free(model->W[l]);
    }
    free(model->W);

    for (int l = 0; l <= model->L; l++) {
        free(model->X[l]);
        free(model->deltas[l]);
    }
    free(model->X);
    free(model->deltas);

    free(model->d);
    free(model);
}