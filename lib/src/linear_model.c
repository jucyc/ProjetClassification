#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/linear_model.h"

static void init_weights(float** weights, int n_classes, int n_weights_per_class) {
    for (int c = 0; c < n_classes; c++) {
        for (int j = 0; j < n_weights_per_class; j++) {
            weights[c][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

static float dot_with_bias(float* w, float* x, int n_features) {
    float score = w[0];
    for (int j = 0; j < n_features; j++) {
        score += w[j + 1] * x[j];
    }
    return score;
}

static float sign_output(float score) {
    return (score >= 0.0f) ? 1.0f : -1.0f;
}

LinearModel* linear_create(int n_features, int n_classes) {
    LinearModel* model = (LinearModel*)malloc(sizeof(LinearModel));
    model->n_features = n_features;
    model->n_classes = n_classes;
    model->is_trained = 0;

    int n_weights_per_class = n_features + 1;

    model->weights = (float**)malloc(n_classes * sizeof(float*));
    for (int c = 0; c < n_classes; c++) {
        model->weights[c] = (float*)malloc(n_weights_per_class * sizeof(float));
    }

    init_weights(model->weights, n_classes, n_weights_per_class);

    return model;
}


void linear_train(LinearModel* model, float** X, int* y, int n_samples, float learning_rate, int n_iterations) 
{
    for (int c = 0; c < model->n_classes; c++) {
        float* w = model->weights[c];

        for (int it = 0; it < n_iterations; it++) {
            int k = rand() % n_samples;
            float* xk = X[k];

            float Yk = (y[k] == c) ? 1.0f : -1.0f;
            float score = dot_with_bias(w, xk, model->n_features);
            float gXk = sign_output(score);

            float error = learning_rate * (Yk - gXk);

            w[0] += error;
            for (int j = 0; j < model->n_features; j++) {
                w[j + 1] += error * xk[j];
            }
        }

        printf("Perceptron classe %d entraine (%d iterations)\n", c, n_iterations);
    }

    model->is_trained = 1;
}

/*
 * Prediction multi-classes : on calcule le score W.X de chaque perceptron
 * et on garde la classe dont le perceptron est le plus confiant (score le
 * plus eleve), comme dans le notebook "Multi Linear 3 classes" fourni par
 * Vidal.
 */
int linear_predict(LinearModel* model, float* x) {
    int best_class = 0;
    float best_score = dot_with_bias(model->weights[0], x, model->n_features);

    for (int c = 1; c < model->n_classes; c++) {
        float score = dot_with_bias(model->weights[c], x, model->n_features);
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }

    return best_class;
}

float* linear_predict_scores(LinearModel* model, float* x) {
    float* scores = (float*)malloc(model->n_classes * sizeof(float));
    for (int c = 0; c < model->n_classes; c++) {
        scores[c] = dot_with_bias(model->weights[c], x, model->n_features);
    }
    return scores;
}

void linear_save(LinearModel* model, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    fwrite(&model->n_features, sizeof(int), 1, f);
    fwrite(&model->n_classes, sizeof(int), 1, f);

    int n_weights_per_class = model->n_features + 1;
    for (int c = 0; c < model->n_classes; c++) {
        fwrite(model->weights[c], sizeof(float), n_weights_per_class, f);
    }

    fclose(f);
}

LinearModel* linear_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    int n_features, n_classes;
    fread(&n_features, sizeof(int), 1, f);
    fread(&n_classes, sizeof(int), 1, f);

    LinearModel* model = linear_create(n_features, n_classes);

    int n_weights_per_class = n_features + 1;
    for (int c = 0; c < n_classes; c++) {
        fread(model->weights[c], sizeof(float), n_weights_per_class, f);
    }

    model->is_trained = 1;
    fclose(f);
    return model;
}

void linear_free(LinearModel* model) {
    for (int c = 0; c < model->n_classes; c++) {
        free(model->weights[c]);
    }
    free(model->weights);
    free(model);
}