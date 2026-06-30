#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

typedef struct {
    float** weights;   // [n_classes x (n_features + 1)], +1 => biais
    int n_classes;
    int n_features;
    int is_trained;
} LinearModel;

LinearModel* linear_create(int n_features, int n_classes);

void linear_free(LinearModel* model);

void linear_train(LinearModel* model, float** X, int* y, int n_samples,
                  float learning_rate, int n_iterations);

int linear_predict(LinearModel* model, float* x);

float* linear_predict_scores(LinearModel* model, float* x);

void linear_save(LinearModel* model, const char* filename);
LinearModel* linear_load(const char* filename);

#endif