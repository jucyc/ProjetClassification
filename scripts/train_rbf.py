import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.image_processor import ImageProcessor
from python_api.rbf_bridge import RBFModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns


def plot_confusion_matrix(y_test, y_pred, class_names, save_path='rapport/confusion_matrix_rbf.png'):
    os.makedirs('rapport', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédictions", fontsize=12)
    plt.ylabel("Vérité terrain", fontsize=12)
    plt.title("Matrice de confusion - RBF (pseudo-inverse)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion sauvegardée: {save_path}")


def plot_class_metrics(y_test, y_pred, class_names, save_path='rapport/metrics_by_class_rbf.png'):
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
    plt.title('Métriques par classe - RBF', fontsize=14)
    plt.xticks(x, class_names, rotation=15)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Métriques par classe sauvegardées: {save_path}")


def plot_learning_curve(X_train, y_train, X_test, y_test,
                        n_features, n_classes, gamma,
                        save_path='rapport/learning_curve_rbf.png'):
    """
    Courbe d'apprentissage du RBF : on fait varier le nombre de centres K
    et on mesure l'accuracy train/test. Le RBF n'a pas d'iterations
    d'apprentissage (formule fermee), donc on fait varier K plutot que
    le nombre d'iterations.
    """
    os.makedirs('rapport', exist_ok=True)

    paliers_k = [5, 10, 20, 30, 50, 80]
    acc_train = []
    acc_test = []

    for k in paliers_k:
        m = RBFModel(n_centers=k, n_features=n_features,
                     n_classes=n_classes, gamma=gamma)
        m.train(X_train, y_train, n_iter=100)

        preds_train = np.array([m.predict(x) for x in X_train])
        preds_test  = np.array([m.predict(x) for x in X_test])

        acc_train.append(float(np.mean(preds_train == np.array(y_train))))
        acc_test.append(float(np.mean(preds_test  == np.array(y_test))))

        print(f"  K={k:>3} centres -> train={acc_train[-1]:.2%}, test={acc_test[-1]:.2%}")

    plt.figure(figsize=(8, 5))
    plt.plot(paliers_k, acc_train, 'o-', color='#3498db', label='Train')
    plt.plot(paliers_k, acc_test,  'o-', color='#e74c3c', label='Test')
    plt.xlabel("Nombre de centres K")
    plt.ylabel("Accuracy")
    plt.title("Courbe d'apprentissage — RBF (impact du nombre de centres)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Courbe d'apprentissage sauvegardée: {save_path}")


def generate_all_graphs(y_test, y_pred, class_names,
                        X_train=None, y_train=None, X_test=None,
                        n_features=None, n_classes=None, gamma=None):
    print("\n" + "="*50)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("="*50)

    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_class_metrics(y_test, y_pred, class_names)

    if X_train is not None:
        print("\nGénération courbe d'apprentissage (teste 6 valeurs de K, patience)...")
        plot_learning_curve(X_train, y_train, X_test, y_test,
                            n_features, n_classes, gamma)

    print("\nTous les graphiques ont été générés dans le dossier 'rapport/'")


def train_rbf_model(n_centers=30, gamma=0.001, n_iter=100):
    """
    n_centers : nombre de centres K (representants du dataset)
    gamma     : parametre de largeur des gaussiennes (slide p.103)
                plus gamma est grand, plus les zones d'influence sont etroites
    n_iter    : nombre d'iterations pour k-means (Lloyd)
    """
    print("="*60)
    print("ENTRAÎNEMENT DU RBF (Radial Basis Function)")
    print("="*60)

    processor = ImageProcessor()
    X, y = processor.load_dataset('data/raw', normalize=True)
    np.save('data/models/rbf_mean.npy', processor.mean)
    np.save('data/models/rbf_std.npy', processor.std)
    print("Stats de normalisation sauvegardées.")

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

    print(f"\nParametres: K={n_centers} centres, gamma={gamma}")
    model = RBFModel(n_centers=n_centers, n_features=X.shape[1],
                     n_classes=n_classes, gamma=gamma)

    print("\nEntraînement en cours (k-means + pseudo-inverse, notre lib C)...")
    model.train(X_train.tolist(), y_train.tolist(), n_iter=n_iter)

    y_pred = np.array([model.predict(x) for x in X_test.tolist()])
    accuracy = np.mean(y_pred == y_test)

    print(f"\nPrécision sur le test: {accuracy:.2%}")

    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred,
                                target_names=processor.class_names,
                                zero_division=0))

    os.makedirs('data/models', exist_ok=True)
    model.save('data/models/rbf_model.bin')

    generate_all_graphs(y_test, y_pred, processor.class_names,
                        X_train=X_train.tolist(), y_train=y_train.tolist(),
                        X_test=X_test.tolist(),
                        n_features=X.shape[1], n_classes=n_classes, gamma=gamma)

    return model, accuracy


if __name__ == "__main__":
    train_rbf_model(n_centers=30, gamma=0.001, n_iter=100)