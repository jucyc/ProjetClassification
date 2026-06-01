import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.image_processor import ImageProcessor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns

class LinearModel:
    def __init__(self, n_features=52, n_classes=3, learning_rate=0.1):
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weights = np.random.randn(n_classes, n_features) * 0.01
        self.biases = np.zeros(n_classes)
        self.losses = []
    
    def softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def train(self, X, y, epochs=1000, verbose=1):
        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)
        
        for epoch in range(epochs):
            scores = np.dot(X, self.weights.T) + self.biases
            probs = self.softmax(scores)
            
            correct_log_probs = -np.log(probs[np.arange(n_samples), y] + 1e-8)
            loss = np.mean(correct_log_probs)
            self.losses.append(loss)
            
            dscores = probs
            dscores[np.arange(n_samples), y] -= 1
            dscores /= n_samples
            
            dW = np.dot(dscores.T, X)
            db = np.sum(dscores, axis=0)
            
            self.weights -= self.learning_rate * dW
            self.biases -= self.learning_rate * db
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, x):
        x = np.array(x)
        scores = np.dot(self.weights, x) + self.biases
        return np.argmax(scores)
    
    def predict_batch(self, X):
        X = np.array(X)
        scores = np.dot(X, self.weights.T) + self.biases
        return np.argmax(scores, axis=1)
    
    def save(self, filename):
        np.savez(filename, weights=self.weights, biases=self.biases, losses=self.losses)
        print(f"Modèle sauvegardé: {filename}")

# ==================== FONCTIONS DE GRAPHIQUES ====================

def plot_loss_curve(model, save_path='rapport/loss_curve.png'):
    """Génère la courbe d'apprentissage"""
    os.makedirs('rapport', exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Sous-graphique 1: loss normale
    plt.subplot(1, 2, 1)
    plt.plot(model.losses, linewidth=2, color='blue')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
    plt.title("Courbe d'apprentissage", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=model.losses[-1], color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'Loss finale: {model.losses[-1]:.4f}', 
                 xy=(len(model.losses)-100, model.losses[-1]),
                 fontsize=10, color='red')
    
    # Sous-graphique 2: loss en échelle logarithmique
    plt.subplot(1, 2, 2)
    plt.semilogy(model.losses, linewidth=2, color='green')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (échelle log)", fontsize=12)
    plt.title("Courbe d'apprentissage (vue log)", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Courbe de loss sauvegardée: {save_path}")

def plot_confusion_matrix(y_test, y_pred, class_names, save_path='rapport/confusion_matrix.png'):
    """Génère la matrice de confusion"""
    os.makedirs('rapport', exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédictions", fontsize=12)
    plt.ylabel("Vérité terrain", fontsize=12)
    plt.title("Matrice de confusion", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion sauvegardée: {save_path}")

def plot_class_metrics(y_test, y_pred, class_names, save_path='rapport/metrics_by_class.png'):
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
    plt.title('Métriques par classe', fontsize=14)
    plt.xticks(x, class_names, rotation=15)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Métriques par classe sauvegardées: {save_path}")

def plot_class_distribution(y, class_names, save_path='rapport/class_distribution.png'):
    """Distribution des classes"""
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

def generate_all_graphs(model, X_test, y_test, y_pred, class_names, y_all):
    """Génère tous les graphiques en une seule fois"""
    print("\n" + "="*50)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("="*50)
    
    # 1. Courbe de loss
    plot_loss_curve(model)
    
    # 2. Matrice de confusion
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    # 3. Métriques par classe
    plot_class_metrics(y_test, y_pred, class_names)
    
    # 4. Distribution des classes
    plot_class_distribution(y_all, class_names)
    
    print("\nTous les graphiques ont été générés dans le dossier 'rapport/'")

def train_linear_model():
    print("="*60)
    print("ENTRAÎNEMENT DU MODÈLE LINÉAIRE")
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
    
    # Split train/test
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\nTrain: {len(X_train)} images")
    print(f"Test: {len(X_test)} images")
    
    # Entraînement
    model = LinearModel(n_features=X.shape[1], n_classes=3, learning_rate=0.1)
    
    print("\nEntraînement en cours...")
    model.train(X_train, y_train, epochs=1000, verbose=1)
    
    # Évaluation
    y_pred = model.predict_batch(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\nPrécision sur le test: {accuracy:.2%}")
    
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, target_names=processor.class_names, zero_division=0))
    
    # Sauvegarde
    os.makedirs('data/models', exist_ok=True)
    model.save('data/models/linear_model_pure.npz')
    
    # Générer les graphiques
    generate_all_graphs(model, X_test, y_test, y_pred, processor.class_names, y)
    
    return model, accuracy

if __name__ == "__main__":
    train_linear_model()