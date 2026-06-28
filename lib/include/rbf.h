#ifndef RBF_H
#define RBF_H

typedef struct {
    int input_size;
    int n_centers;
    int output_size;
    double gamma;        // paramètre de largeur de la gaussienne
    double *centers;      // [n_centers x input_size]
    double *W;            // poids couche de sortie [n_centers x output_size]
    double *b;            // biais couche de sortie [output_size]
} RBF;

RBF* rbf_create(int input_size, int n_centers, int output_size, double gamma);
void rbf_destroy(RBF* net);

// Étape 1 : choisir les centres avec k-means sur les données d'entraînement
void rbf_fit_centers(RBF* net, double* X, int n_samples, int kmeans_iters);

// Étape 2 : entraîner la couche de sortie (descente de gradient)
void rbf_train(RBF* net, double* X, int* y, int n_samples,
               double lr, int epochs, int batch_size);

int rbf_predict(RBF* net, double* x);
double rbf_evaluate(RBF* net, double* X, int* y, int n_samples);

void rbf_save(RBF* net, const char* path);
RBF* rbf_load(const char* path);

#endif