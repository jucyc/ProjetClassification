#ifndef MLP_H
#define MLP_H

typedef struct {
    int* d;          
    int n_layers;     
    int L;             //(L = n_layers - 1)
    double*** W;
    double** X;        
    double** deltas;
    int is_trained;
} MLP;

MLP* mlp_create(int* npl, int n_layers);
void mlp_free(MLP* model);

void mlp_train(MLP* model, double** X, double** Y, int n_samples,
               double learning_rate, int n_iterations);

double* mlp_predict_raw(MLP* model, double* x);

int mlp_predict_class(MLP* model, double* x);

void mlp_save(MLP* model, const char* filename);
MLP* mlp_load(const char* filename);

#endif