import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.image_processor import ImageProcessor
from python_api.ml_bridge import MLPModel
from sklearn.metrics import classification_report


def one_hot(labels, n_classes):
    out = []
    for label in labels:
        row = [-1.0] * n_classes
        row[label] = 1.0
        out.append(row)
    return out


def train_mlp_model(hidden_layers=(32,), n_iterations=300000, learning_rate=0.01):
    print("="*60)
    print("ENTRAÎNEMENT DU MLP (Perceptron Multi-Couches)")
    print("="*60)

    processor = ImageProcessor()
    X, y = processor.load_dataset('data/raw', normalize=True)

    if len(X) == 0:
        print("Aucune image trouvée!")
        return None, 0

    n_classes = 3
    print(f"\nDataset: {len(X)} images, {X.shape[1]} features")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   {processor.class_names[cls]}: {count} images")

    np.random.seed(42)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain: {len(X_train)} images")
    print(f"Test: {len(X_test)} images")

    npl = [X.shape[1]] + list(hidden_layers) + [n_classes]
    print(f"\nArchitecture du MLP: {npl}")

    model = MLPModel(npl)
    print(f"\nEntraînement en cours (notre lib C, retropropagation Tanh, {n_iterations} iterations)...")
    Y_train_onehot = one_hot(y_train.tolist(), n_classes)
    model.train(X_train.tolist(), Y_train_onehot,
                learning_rate=learning_rate, n_iterations=n_iterations)

    y_pred = np.array([model.predict(x) for x in X_test.tolist()])
    accuracy = np.mean(y_pred == y_test)

    print(f"\nPrécision sur le test: {accuracy:.2%}")
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred,
                                target_names=processor.class_names,
                                zero_division=0))

    os.makedirs('data/models', exist_ok=True)
    model.save('data/models/mlp_model.bin')

    return model, accuracy


if __name__ == "__main__":
    train_mlp_model(hidden_layers=(32,), n_iterations=300000, learning_rate=0.01)