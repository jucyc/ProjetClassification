import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.image_processor import ImageProcessor
from python_api.ml_bridge import LinearModel
from sklearn.metrics import classification_report


def train_linear_model():
    print("="*60)
    print("ENTRAÎNEMENT DU MODÈLE LINÉAIRE (Perceptron / Rosenblatt)")
    print("="*60)

    processor = ImageProcessor()
    X, y = processor.load_dataset('data/raw', normalize=True)

    if len(X) == 0:
        print("Aucune image trouvée!")
        return None, 0

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

    model = LinearModel(n_features=X.shape[1], n_classes=3)
    print("\nEntraînement en cours (notre lib C, regle de Rosenblatt)...")
    model.train(X_train.tolist(), y_train.tolist(),
                learning_rate=0.01, n_iterations=50000)

    y_pred = np.array([model.predict(x) for x in X_test.tolist()])
    accuracy = np.mean(y_pred == y_test)

    print(f"\nPrécision sur le test: {accuracy:.2%}")
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred,
                                target_names=processor.class_names,
                                zero_division=0))

    os.makedirs('data/models', exist_ok=True)
    model.save('data/models/linear_model.bin')

    return model, accuracy


if __name__ == "__main__":
    train_linear_model()