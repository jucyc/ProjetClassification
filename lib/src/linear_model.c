#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/linear_model.h"

/*
 * Implementation : Perceptron(s) + regle de Rosenblatt, one-vs-rest.
 * Voir linear_model.h pour les explications detaillees et les references
 * au cours.
 *
 * Convention interne : le biais est stocke comme un poids supplementaire
 * weights[c][0], applique a une entree fictive x0 = 1 (comme dans les
 * slides : "Xk les parametres de l'exemple k ET le biais x0_k = 1").
 * Donc weights[c] a (n_features + 1) cases :
 *   weights[c][0]            -> biais
 *   weights[c][1..n_features] -> poids des vraies features
 */

// Initialisation aleatoire des poids dans [-1, 1], comme demande par les slides
static void init_weights(float** weights, int n_classes, int n_weights_per_class) {
    for (int c = 0; c < n_classes; c++) {
        for (int j = 0; j < n_weights_per_class; j++) {
            weights[c][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

// Score brut W.X pour un perceptron donne (x ne contient PAS le biais,
// on l'ajoute nous-memes ici)
static float dot_with_bias(float* w, float* x, int n_features) {
    float score = w[0]; // biais * 1
    for (int j = 0; j < n_features; j++) {
        score += w[j + 1] * x[j];
    }
    return score;
}

// g(x) = signe(W.X), sortie en -1 / +1 (cf slides p.62-65)
static float sign_output(float score) {
    return (score >= 0.0f) ? 1.0f : -1.0f;
}

LinearModel* linear_create(int n_features, int n_classes) {
    LinearModel* model = (LinearModel*)malloc(sizeof(LinearModel));
    model->n_features = n_features;
    model->n_classes = n_classes;
    model->is_trained = 0;

    int n_weights_per_class = n_features + 1; // +1 pour le biais

    model->weights = (float**)malloc(n_classes * sizeof(float*));
    for (int c = 0; c < n_classes; c++) {
        model->weights[c] = (float*)malloc(n_weights_per_class * sizeof(float));
    }

    init_weights(model->weights, n_classes, n_weights_per_class);

    return model;
}

/*
 * Entrainement par la regle de Rosenblatt, en one-vs-rest.
 *
 * Pour chaque perceptron c (un par classe) :
 *   On repete n_iterations fois :
 *     - tirer un exemple k au hasard
 *     - Yk = +1 si l'exemple k appartient a la classe c, sinon -1
 *     - calculer g(Xk) = signe(W_c . Xk)
 *     - W_c <- W_c + alpha * (Yk - g(Xk)) * Xk   (avec x0 = 1 pour le biais)
 *
 * C'est exactement la regle du notebook d'exemple fourni par Vidal
 * (_Exemple__Classification_Linéaire_en_python), appliquee une fois par
 * classe pour gerer le multi-classes.
 */
void linear_train(LinearModel* model, float** X, int* y, int n_samples,
                  float learning_rate, int n_iterations) {
    for (int c = 0; c < model->n_classes; c++) {
        float* w = model->weights[c];

        for (int it = 0; it < n_iterations; it++) {
            int k = rand() % n_samples;
            float* xk = X[k];

            float Yk = (y[k] == c) ? 1.0f : -1.0f;
            float score = dot_with_bias(w, xk, model->n_features);
            float gXk = sign_output(score);

            float error = learning_rate * (Yk - gXk);

            // Mise a jour du biais (x0 = 1) puis des vrais poids
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