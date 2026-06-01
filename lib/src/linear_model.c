#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/linear_model.h"

// Initialisation aléatoire des poids
static void init_weights(float** weights, int n_classes, int n_features) {
    for (int i = 0; i < n_classes; i++) {
        for (int j = 0; j < n_features; j++) {
            weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
    }
}

// Fonction softmax
static void softmax(float* scores, float* probs, int n_classes) {
    float max_score = scores[0];
    for (int i = 1; i < n_classes; i++) {
        if (scores[i] > max_score) max_score = scores[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n_classes; i++) {
        probs[i] = expf(scores[i] - max_score);
        sum += probs[i];
    }
    
    for (int i = 0; i < n_classes; i++) {
        probs[i] /= sum;
    }
}

LinearModel* linear_create(int n_features, int n_classes) {
    LinearModel* model = (LinearModel*)malloc(sizeof(LinearModel));
    model->n_features = n_features;
    model->n_classes = n_classes;
    model->is_trained = 0;
    
    // Allocation des poids
    model->weights = (float**)malloc(n_classes * sizeof(float*));
    for (int i = 0; i < n_classes; i++) {
        model->weights[i] = (float*)malloc(n_features * sizeof(float));
    }
    
    model->biases = (float*)malloc(n_classes * sizeof(float));
    
    init_weights(model->weights, n_classes, n_features);
    memset(model->biases, 0, n_classes * sizeof(float));
    
    return model;
}

void linear_train(LinearModel* model, float** X, int* y, int n_samples,
                  float learning_rate, int epochs) {
    float* scores = (float*)malloc(model->n_classes * sizeof(float));
    float* probs = (float*)malloc(model->n_classes * sizeof(float));
    float* gradients_w = (float*)calloc(model->n_classes * model->n_features, sizeof(float));
    float* gradients_b = (float*)calloc(model->n_classes, sizeof(float));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Remettre les gradients à zéro
        memset(gradients_w, 0, model->n_classes * model->n_features * sizeof(float));
        memset(gradients_b, 0, model->n_classes * sizeof(float));
        
        float total_loss = 0.0f;
        
        // Calcul des gradients sur tout le batch
        for (int i = 0; i < n_samples; i++) {
            // Forward pass
            for (int c = 0; c < model->n_classes; c++) {
                scores[c] = model->biases[c];
                for (int j = 0; j < model->n_features; j++) {
                    scores[c] += model->weights[c][j] * X[i][j];
                }
            }
            
            softmax(scores, probs, model->n_classes);
            
            // Cross-entropy loss
            total_loss += -logf(probs[y[i]]);
            
            // Backward pass
            for (int c = 0; c < model->n_classes; c++) {
                float error = probs[c] - (c == y[i] ? 1.0f : 0.0f);
                gradients_b[c] += error;
                for (int j = 0; j < model->n_features; j++) {
                    gradients_w[c * model->n_features + j] += error * X[i][j];
                }
            }
        }
        
        // Mise à jour des poids
        for (int c = 0; c < model->n_classes; c++) {
            model->biases[c] -= learning_rate * gradients_b[c] / n_samples;
            for (int j = 0; j < model->n_features; j++) {
                model->weights[c][j] -= learning_rate * gradients_w[c * model->n_features + j] / n_samples;
            }
        }
        
        // Affichage de la loss tous les 100 epochs
        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %.4f\n", epoch, total_loss / n_samples);
        }
    }
    
    model->is_trained = 1;
    
    free(scores);
    free(probs);
    free(gradients_w);
    free(gradients_b);
}

int linear_predict(LinearModel* model, float* x) {
    float* scores = (float*)malloc(model->n_classes * sizeof(float));
    
    for (int c = 0; c < model->n_classes; c++) {
        scores[c] = model->biases[c];
        for (int j = 0; j < model->n_features; j++) {
            scores[c] += model->weights[c][j] * x[j];
        }
    }
    
    int pred_class = 0;
    float max_score = scores[0];
    for (int c = 1; c < model->n_classes; c++) {
        if (scores[c] > max_score) {
            max_score = scores[c];
            pred_class = c;
        }
    }
    
    free(scores);
    return pred_class;
}

float* linear_predict_proba(LinearModel* model, float* x) {
    float* scores = (float*)malloc(model->n_classes * sizeof(float));
    float* probs = (float*)malloc(model->n_classes * sizeof(float));
    
    for (int c = 0; c < model->n_classes; c++) {
        scores[c] = model->biases[c];
        for (int j = 0; j < model->n_features; j++) {
            scores[c] += model->weights[c][j] * x[j];
        }
    }
    
    softmax(scores, probs, model->n_classes);
    free(scores);
    return probs;
}

void linear_save(LinearModel* model, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fwrite(&model->n_features, sizeof(int), 1, f);
    fwrite(&model->n_classes, sizeof(int), 1, f);
    
    for (int i = 0; i < model->n_classes; i++) {
        fwrite(model->weights[i], sizeof(float), model->n_features, f);
    }
    fwrite(model->biases, sizeof(float), model->n_classes, f);
    
    fclose(f);
}

LinearModel* linear_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    
    int n_features, n_classes;
    fread(&n_features, sizeof(int), 1, f);
    fread(&n_classes, sizeof(int), 1, f);
    
    LinearModel* model = linear_create(n_features, n_classes);
    
    for (int i = 0; i < n_classes; i++) {
        fread(model->weights[i], sizeof(float), n_features, f);
    }
    fread(model->biases, sizeof(float), n_classes, f);
    
    model->is_trained = 1;
    fclose(f);
    return model;
}

void linear_free(LinearModel* model) {
    for (int i = 0; i < model->n_classes; i++) {
        free(model->weights[i]);
    }
    free(model->weights);
    free(model->biases);
    free(model);
}