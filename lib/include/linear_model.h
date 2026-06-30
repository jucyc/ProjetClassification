#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

/*
 * Modele lineaire de classification = Perceptron(s), regle de Rosenblatt.
 *
 * Cours (ESGI - Vidal, slides "Apprendre - Modele Lineaire et PMC", p.62-65) :
 *   - Un perceptron simple separe 2 classes avec une sortie en -1 / +1 :
 *         out = sign(W . X)   (avec biais inclus dans X, x0 = 1)
 *   - Regle de Rosenblatt (marche pour sorties a -1/+1 ou 0/1) :
 *         W <- W + alpha * (Yk - g(Xk)) * Xk
 *     On tire un exemple k au hasard, on regarde l'erreur entre la sortie
 *     attendue Yk et la sortie obtenue g(Xk), et on deplace W un petit peu
 *     dans la bonne direction. On repete plein de fois.
 *
 * Pour gerer plus de 2 classes (notre cas : 3 monuments), on utilise une
 * strategie "one-vs-rest" : un perceptron par classe (n_classes
 * perceptrons), chacun entraine a repondre "+1 si c'est ma classe, -1
 * sinon". A la prediction, on garde le perceptron le plus confiant
 * (le score W.X le plus grand), exactement comme le notebook de cas de
 * tests fourni par Vidal ("Multi Linear 3 classes : Linear Model x3").
 *
 * Pas de regression ici : seule la classification est implementee, car
 * notre probleme applicatif (classer 3 monuments) est un probleme de
 * classification.
 */

typedef struct {
    float** weights;   // [n_classes x (n_features + 1)], le +1 est le biais (poids 0)
    int n_classes;
    int n_features;
    int is_trained;
} LinearModel;

// Création et destruction
LinearModel* linear_create(int n_features, int n_classes);
void linear_free(LinearModel* model);

// Entraînement : regle de Rosenblatt, one-vs-rest, n_iterations tirages aleatoires
void linear_train(LinearModel* model, float** X, int* y, int n_samples,
                  float learning_rate, int n_iterations);

// Prédiction : classe = perceptron qui donne le score le plus eleve
int linear_predict(LinearModel* model, float* x);

// Scores bruts (W.X) de chaque perceptron, utile pour debug / confiance
// relative (ce n'est PAS une probabilite, contrairement a un softmax)
float* linear_predict_scores(LinearModel* model, float* x);

// Sauvegarde et chargement
void linear_save(LinearModel* model, const char* filename);
LinearModel* linear_load(const char* filename);

#endif