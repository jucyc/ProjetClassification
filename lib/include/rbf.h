#ifndef RBF_H
#define RBF_H

typedef struct {
    double** centers;   // [n_centers x n_features] : les K centres (mu)
    double** W;         // [n_centers x n_classes]  : les poids de sortie
    int n_centers;      // K : nombre de centres
    int n_features;     // dimension de chaque exemple
    int n_classes;      // nombre de classes (3 pour nos monuments)
    double gamma;       // parametre de largeur des fonctions gaussiennes
    int is_trained;
} RBFModel;

RBFModel* rbf_create(int n_centers, int n_features, int n_classes, double gamma);
void rbf_free(RBFModel* model);

void rbf_train(RBFModel* model, double** X, int* y, int n_samples, int n_iter);

int rbf_predict(RBFModel* model, double* x);
double* rbf_predict_scores(RBFModel* model, double* x);

void rbf_save(RBFModel* model, const char* filename);
RBFModel* rbf_load(const char* filename);

#endif