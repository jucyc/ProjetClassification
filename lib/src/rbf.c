#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/rbf.h"

static double dist_sq(double* a, double* b, int n) {
    double d = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

static double rbf_kernel(double* x, double* center, int n, double gamma) {
    return exp(-gamma * dist_sq(x, center, n));
}

static void kmeans(double** X, int n_samples, int n_features,
                   double** centers, int n_centers, int n_iter) {
    int* assignments = (int*)malloc(n_samples * sizeof(int));
    int* counts      = (int*)malloc(n_centers * sizeof(int));

    for (int k = 0; k < n_centers; k++) {
        memcpy(centers[k], X[k], n_features * sizeof(double));
    }

    for (int iter = 0; iter < n_iter; iter++) {
        for (int i = 0; i < n_samples; i++) {
            int best_k = 0;
            double best_d = dist_sq(X[i], centers[0], n_features);
            for (int k = 1; k < n_centers; k++) {
                double d = dist_sq(X[i], centers[k], n_features);
                if (d < best_d) { best_d = d; best_k = k; }
            }
            assignments[i] = best_k;
        }

        for (int k = 0; k < n_centers; k++) {
            memset(centers[k], 0, n_features * sizeof(double));
            counts[k] = 0;
        }
        for (int i = 0; i < n_samples; i++) {
            int k = assignments[i];
            counts[k]++;
            for (int f = 0; f < n_features; f++) {
                centers[k][f] += X[i][f];
            }
        }
        for (int k = 0; k < n_centers; k++) {
            if (counts[k] > 0) {
                for (int f = 0; f < n_features; f++) {
                    centers[k][f] /= counts[k];
                }
            }
        }
    }

    free(assignments);
    free(counts);
}

static void compute_weights(double** phi, double** Y,
                             int N, int K, int C, double** W) {
    double** A = (double**)malloc(K * sizeof(double*));
    for (int i = 0; i < K; i++) {
        A[i] = (double*)calloc(K, sizeof(double));
        for (int j = 0; j < K; j++) {
            for (int n = 0; n < N; n++) {
                A[i][j] += phi[n][i] * phi[n][j];
            }
        }
        A[i][i] += 1e-6; 
    }

    double** B = (double**)malloc(K * sizeof(double*));
    for (int i = 0; i < K; i++) {
        B[i] = (double*)calloc(C, sizeof(double));
        for (int c = 0; c < C; c++) {
            for (int n = 0; n < N; n++) {
                B[i][c] += phi[n][i] * Y[n][c];
            }
        }
    }

    double** Aug = (double**)malloc(K * sizeof(double*));
    for (int i = 0; i < K; i++) {
        Aug[i] = (double*)malloc((K + C) * sizeof(double));
        memcpy(Aug[i], A[i], K * sizeof(double));
        memcpy(Aug[i] + K, B[i], C * sizeof(double));
    }

    for (int col = 0; col < K; col++) {
        int pivot = col;
        for (int row = col + 1; row < K; row++) {
            if (fabs(Aug[row][col]) > fabs(Aug[pivot][col])) pivot = row;
        }
        double* tmp = Aug[col]; Aug[col] = Aug[pivot]; Aug[pivot] = tmp;

        double diag = Aug[col][col];
        if (fabs(diag) < 1e-12) continue;

        for (int j = col; j < K + C; j++) Aug[col][j] /= diag;

        for (int row = 0; row < K; row++) {
            if (row == col) continue;
            double factor = Aug[row][col];
            for (int j = col; j < K + C; j++) {
                Aug[row][j] -= factor * Aug[col][j];
            }
        }
    }

    for (int i = 0; i < K; i++) {
        for (int c = 0; c < C; c++) W[i][c] = Aug[i][K + c];
        free(Aug[i]);
    }
    free(Aug);

    for (int i = 0; i < K; i++) { free(A[i]); free(B[i]); }
    free(A); free(B);
}


RBFModel* rbf_create(int n_centers, int n_features, int n_classes, double gamma) {
    RBFModel* model = (RBFModel*)malloc(sizeof(RBFModel));
    model->n_centers  = n_centers;
    model->n_features = n_features;
    model->n_classes  = n_classes;
    model->gamma      = gamma;
    model->is_trained = 0;

    model->centers = (double**)malloc(n_centers * sizeof(double*));
    for (int k = 0; k < n_centers; k++)
        model->centers[k] = (double*)calloc(n_features, sizeof(double));

    model->W = (double**)malloc(n_centers * sizeof(double*));
    for (int k = 0; k < n_centers; k++)
        model->W[k] = (double*)calloc(n_classes, sizeof(double));

    return model;
}

void rbf_free(RBFModel* model) {
    for (int k = 0; k < model->n_centers; k++) {
        free(model->centers[k]);
        free(model->W[k]);
    }
    free(model->centers);
    free(model->W);
    free(model);
}


void rbf_train(RBFModel* model, double** X, int* y, int n_samples, int n_iter) {
    int K = model->n_centers;
    int N = n_samples;
    int C = model->n_classes;
    int F = model->n_features;

    printf("RBF : k-means (%d centres, %d iterations)...\n", K, n_iter);
    kmeans(X, N, F, model->centers, K, n_iter);

    double** phi = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        phi[i] = (double*)malloc(K * sizeof(double));
        for (int k = 0; k < K; k++)
            phi[i][k] = rbf_kernel(X[i], model->centers[k], F, model->gamma);
    }

    double** Y = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        Y[i] = (double*)malloc(C * sizeof(double));
        for (int c = 0; c < C; c++)
            Y[i][c] = (y[i] == c) ? 1.0 : -1.0;
    }

    printf("RBF : calcul des poids par pseudo-inverse...\n");
    compute_weights(phi, Y, N, K, C, model->W);

    for (int i = 0; i < N; i++) { free(phi[i]); free(Y[i]); }
    free(phi); free(Y);

    model->is_trained = 1;
    printf("RBF entraine (%d centres, gamma=%.4f)\n", K, model->gamma);
}

double* rbf_predict_scores(RBFModel* model, double* x) {
    double* scores = (double*)calloc(model->n_classes, sizeof(double));
    for (int k = 0; k < model->n_centers; k++) {
        double phi_k = rbf_kernel(x, model->centers[k],
                                   model->n_features, model->gamma);
        for (int c = 0; c < model->n_classes; c++)
            scores[c] += model->W[k][c] * phi_k;
    }
    return scores;
}

int rbf_predict(RBFModel* model, double* x) {
    double* scores = rbf_predict_scores(model, x);
    int best = 0;
    for (int c = 1; c < model->n_classes; c++)
        if (scores[c] > scores[best]) best = c;
    free(scores);
    return best;
}

void rbf_save(RBFModel* model, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fwrite(&model->n_centers,  sizeof(int),    1, f);
    fwrite(&model->n_features, sizeof(int),    1, f);
    fwrite(&model->n_classes,  sizeof(int),    1, f);
    fwrite(&model->gamma,      sizeof(double), 1, f);
    for (int k = 0; k < model->n_centers; k++)
        fwrite(model->centers[k], sizeof(double), model->n_features, f);
    for (int k = 0; k < model->n_centers; k++)
        fwrite(model->W[k], sizeof(double), model->n_classes, f);
    fclose(f);
}

RBFModel* rbf_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    int n_centers, n_features, n_classes;
    double gamma;
    fread(&n_centers,  sizeof(int),    1, f);
    fread(&n_features, sizeof(int),    1, f);
    fread(&n_classes,  sizeof(int),    1, f);
    fread(&gamma,      sizeof(double), 1, f);
    RBFModel* model = rbf_create(n_centers, n_features, n_classes, gamma);
    for (int k = 0; k < n_centers; k++)
        fread(model->centers[k], sizeof(double), n_features, f);
    for (int k = 0; k < n_centers; k++)
        fread(model->W[k], sizeof(double), n_classes, f);
    model->is_trained = 1;
    fclose(f);
    return model;
}