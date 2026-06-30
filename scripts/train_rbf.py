"""
Entraînement du RBF Network sur le dataset monuments
(Taj Mahal, Grande Muraille, Christ Rédempteur).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from python_api.rbf_bridge import RBF

CLASS_NAMES = ['Taj Mahal', 'Grande Muraille', 'Christ Rédempteur']


def main():
    print("=" * 60)
    print("ENTRAINEMENT RBF SUR LE DATASET MONUMENTS")
    print("=" * 60)

    X_train = np.load('test_cases/X_train.npy')
    y_train = np.load('test_cases/y_train.npy')
    X_test = np.load('test_cases/X_test.npy')
    y_test = np.load('test_cases/y_test.npy')

    print(f"\nTrain : {X_train.shape[0]} images, {X_train.shape[1]} features")
    print(f"Test  : {X_test.shape[0]} images")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name} : {np.sum(y_train == i)} (train) / {np.sum(y_test == i)} (test)")

    n_features = X_train.shape[1]
    n_centers = 150  
    gamma = 0.0005  

    print(f"\nRBF [{n_features} features -> {n_centers} centres -> 3 classes], gamma={gamma}")

    model = RBF(input_size=n_features, n_centers=n_centers, output_size=3, gamma=gamma)

    print("\nEtape 1 : k-means pour trouver les centres...")
    t0 = time.time()
    model.fit_centers(X_train.tolist(), kmeans_iters=15)
    print(f"  (termine en {time.time()-t0:.1f}s)")

    print("\nEtape 2 : entrainement de la couche de sortie...")
    t0 = time.time()
    model.train(X_train.tolist(), y_train.tolist(),
                learning_rate=0.1, epochs=800, batch_size=len(X_train))
    print(f"  (termine en {time.time()-t0:.1f}s)")

    acc_train = model.evaluate(X_train.tolist(), y_train.tolist())
    acc_test = model.evaluate(X_test.tolist(), y_test.tolist())

    print(f"\nPrecision train : {acc_train:.2f}%")
    print(f"Precision test  : {acc_test:.2f}%")

    os.makedirs('data/models', exist_ok=True)
    model.save('data/models/rbf_monuments.bin')

    print("\nTermine !")


if __name__ == "__main__":
    main()