#ifndef MLP_H
#define MLP_H

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double *W1;
    double *b1;
    double *W2;
    double *b2;
} MLP;

MLP* mlp_create(int input_size, int hidden_size, int output_size);
void mlp_destroy(MLP* net);
void mlp_train(MLP* net, double* X, int* y, int n_samples, 
               double lr, int epochs, int batch_size);
int mlp_predict(MLP* net, double* x);
double mlp_evaluate(MLP* net, double* X, int* y, int n_samples);
void mlp_save(MLP* net, const char* path);
MLP* mlp_load(const char* path);

#endif