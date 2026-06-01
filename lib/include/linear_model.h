#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

typedef struct {
    float** weights;  // [n_classes x n_features]
    float* biases;    // [n_classes]
    int n_classes;
    int n_features;
    int is_trained;
} LinearModel;

// Création et destruction
LinearModel* linear_create(int n_features, int n_classes);
void linear_free(LinearModel* model);

// Entraînement
void linear_train(LinearModel* model, float** X, int* y, int n_samples, 
                  float learning_rate, int epochs);

// Prédiction
int linear_predict(LinearModel* model, float* x);
float* linear_predict_proba(LinearModel* model, float* x);

// Sauvegarde et chargement
void linear_save(LinearModel* model, const char* filename);
LinearModel* linear_load(const char* filename);

#endif