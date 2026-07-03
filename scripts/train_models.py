import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.image_processor import ImageProcessor
from python_api.ml_bridge import LinearModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns

# ==================== FONCTIONS DE GRAPHIQUES ====================

def plot_confusion_matrix(y_test, y_pred, class_names, save_path='rapport/confusion_matrix.png'):
    os.makedirs('rapport', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédictions", fontsize=12)
    plt.ylabel("Vérité terrain", fontsize=12)
    plt.title("Matrice de confusion - Modèle linéaire (Rosenblatt)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion sauvegardée: {save_path}")


def plot_class_metrics(y_test, y_pred, class_names, save_path='rapport/metrics_by_class.png'):
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
    plt.title('Métriques par classe - Modèle linéaire', fontsize=14)
    plt.xticks(x, class_names, rotation=15)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Métriques par classe sauvegardées: {save_path}")


def plot_class_distribution(y, class_names, save_path='rapport/class_distribution.png'):
    os.makedirs('rapport', exist_ok=True)
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_names, counts, color=['#2ecc71', '#e74c3c', '#3498db'])
    plt.ylabel("Nombre d'images", fontsize=12)
    plt.xlabel("Classes", fontsize=12)
    plt.title("Distribution des classes dans le dataset", fontsize=14)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribution des classes sauvegardée: {save_path}")


def plot_learning_curve(X_train, y_train, X_test, y_test, class_names,
                        save_path='rapport/learning_curve_linear.png'):
    """
    Courbe d'apprentissage du modele lineaire : on entraine avec un nombre
    croissant d'iterations et on mesure l'accuracy sur train et test a chaque
    palier. Le Perceptron/Rosenblatt n'a pas de fonction de cout (pas de
    cross-entropy), donc on trace l'accuracy plutot qu'une loss.
    """
    os.makedirs('rapport', exist_ok=True)

    paliers = [1000, 5000, 10000, 20000, 35000, 50000]
    acc_train = []
    acc_test = []

    for n_iter in paliers:
        m = LinearModel(n_features=len(X_train[0]), n_classes=3)
        m.train(X_train, y_train, learning_rate=0.01, n_iterations=n_iter)

        preds_train = np.array([m.predict(x) for x in X_train])
        preds_test  = np.array([m.predict(x) for x in X_test])

        acc_train.append(float(np.mean(preds_train == np.array(y_train))))
        acc_test.append(float(np.mean(preds_test  == np.array(y_test))))

        print(f"  {n_iter:>6} iter -> train={acc_train[-1]:.2%}, test={acc_test[-1]:.2%}")

    plt.figure(figsize=(8, 5))
    plt.plot(paliers, acc_train, 'o-', color='#3498db', label='Train')
    plt.plot(paliers, acc_test,  'o-', color='#e74c3c', label='Test')
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Accuracy")
    plt.title("Courbe d'apprentissage — Modèle linéaire (Rosenblatt)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Courbe d'apprentissage sauvegardée: {save_path}")


def generate_all_graphs(X_test, y_test, y_pred, class_names, y_all,
                        X_train=None, y_train=None):
    print("\n" + "="*50)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("="*50)

    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_class_metrics(y_test, y_pred, class_names)
    plot_class_distribution(y_all, class_names)

    if X_train is not None and y_train is not None:
        print("\nGénération courbe d'apprentissage (entraine 6 fois, patience)...")
        plot_learning_curve(X_train, y_train, X_test, y_test, class_names)

    print("\nTous les graphiques ont été générés dans le dossier 'rapport/'")


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

    print(f"   Stats features: mean={X.mean():.3f}, std={X.std():.3f}")

    # Seed fixee pour que les resultats soient reproductibles d'un run
    # a l'autre (utile en soutenance si Vidal demande de relancer)
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
    n_iterations = 50000
    model.train(X_train.tolist(), y_train.tolist(),
                learning_rate=0.01, n_iterations=n_iterations)

    y_pred = np.array([model.predict(x) for x in X_test.tolist()])
    accuracy = np.mean(y_pred == y_test)

    print(f"\nPrécision sur le test: {accuracy:.2%}")

    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, target_names=processor.class_names, zero_division=0))

    os.makedirs('data/models', exist_ok=True)
    model.save('data/models/linear_model.bin')

    generate_all_graphs(X_test, y_test, y_pred, processor.class_names, y,
                        X_train=X_train.tolist(), y_train=y_train.tolist())

    return model, accuracy


if __name__ == "__main__":
    train_linear_model()