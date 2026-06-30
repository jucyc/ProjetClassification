import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.image_processor import ImageProcessor
from python_api.ml_bridge import MLPModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns

# ==================== FONCTIONS DE GRAPHIQUES ====================

def plot_confusion_matrix(y_test, y_pred, class_names, save_path='rapport/confusion_matrix_mlp.png'):
    """Génère la matrice de confusion"""
    os.makedirs('rapport', exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédictions", fontsize=12)
    plt.ylabel("Vérité terrain", fontsize=12)
    plt.title("Matrice de confusion - MLP (retropropagation Tanh)", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion sauvegardée: {save_path}")


def plot_class_metrics(y_test, y_pred, class_names, save_path='rapport/metrics_by_class_mlp.png'):
    """Diagramme des métriques par classe"""
    os.makedirs('rapport', exist_ok=True)

    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Précision', color='#2ecc71')
    plt.bar(x, recall, width, label='Rappel', color='#3498db')
    plt.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Métriques par classe - MLP', fontsize=14)
    plt.xticks(x, class_names, rotation=15)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Métriques par classe sauvegardées: {save_path}")


def generate_all_graphs(y_test, y_pred, class_names):
    """Génère tous les graphiques en une seule fois"""
    print("\n" + "="*50)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("="*50)

    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_class_metrics(y_test, y_pred, class_names)

    print("\nTous les graphiques ont été générés dans le dossier 'rapport/'")


def one_hot(labels, n_classes):
    """
    Encode des labels entiers en vecteurs one-hot -1.0/+1.0 (PAS 0/1).
    Necessaire car le MLP a plusieurs neurones de sortie en -1/+1
    (contrairement au modele lineaire qui a un perceptron separe par
    classe, traite individuellement en one-vs-rest).
    """
    out = []
    for label in labels:
        row = [-1.0] * n_classes
        row[label] = 1.0
        out.append(row)
    return out


def train_mlp_model(hidden_layers=(32,), n_iterations=300000, learning_rate=0.01):
    """
    hidden_layers : tuple des tailles des couches cachees, ex (32,) pour
    une seule couche cachee de 32 neurones, ou (64, 16) pour deux couches.
    """
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

    print(f"   Stats features: mean={X.mean():.3f}, std={X.std():.3f}")

    # Split train/test (meme seed que train_models.py pour pouvoir
    # comparer les deux modeles sur exactement le meme split)
    np.random.seed(42)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain: {len(X_train)} images")
    print(f"Test: {len(X_test)} images")

    # Architecture : [n_features, couches cachees..., n_classes]
    npl = [X.shape[1]] + list(hidden_layers) + [n_classes]
    print(f"\nArchitecture du MLP: {npl}")

    model = MLPModel(npl)

    print(f"\nEntraînement en cours (notre lib C, retropropagation Tanh, {n_iterations} iterations)...")
    Y_train_onehot = one_hot(y_train.tolist(), n_classes)
    model.train(X_train.tolist(), Y_train_onehot,
                learning_rate=learning_rate, n_iterations=n_iterations)

    # Évaluation
    y_pred = np.array([model.predict(x) for x in X_test.tolist()])
    accuracy = np.mean(y_pred == y_test)

    print(f"\nPrécision sur le test: {accuracy:.2%}")

    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, target_names=processor.class_names, zero_division=0))

    # Sauvegarde
    os.makedirs('data/models', exist_ok=True)
    model.save('data/models/mlp_model.bin')

    # Générer les graphiques
    generate_all_graphs(y_test, y_pred, processor.class_names)

    return model, accuracy


if __name__ == "__main__":
    # Architecture par defaut : 1 couche cachee de 32 neurones.
    # N'hesite pas a essayer d'autres tailles (ex hidden_layers=(64,) ou
    # (64, 16)) pour voir l'impact sur la precision dans le rapport.
    train_mlp_model(hidden_layers=(32,), n_iterations=300000, learning_rate=0.01)