import numpy as np
import os
import sys

def generate_simple_test_data():
    """Génère des données de test simples pour valider les modèles"""
    
    print("="*60)
    print("GÉNÉRATION DES DONNÉES DE TEST")
    print("="*60)
    
    # Test 1: Classification linéaire simple (2 classes)
    print("\nTest 1: Classification linéaire simple (2 classes)")
    X1 = np.array([
        [1, 1], [2, 2], [2, 3], [3, 2],  # Classe 0
        [5, 5], [6, 6], [7, 5], [6, 7],  # Classe 1
    ])
    y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    print(f"   → {len(X1)} échantillons, {X1.shape[1]} features, 2 classes")
    
    # Test 2: XOR (non linéaire)
    print("\nTest 2: XOR (problème non linéaire)")
    X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y2 = np.array([0, 1, 1, 0])
    print(f"   → {len(X2)} échantillons, {X2.shape[1]} features, XOR pattern")
    
    # Test 3: Classification 3 classes aléatoire
    print("\nTest 3: Classification 3 classes (aléatoire)")
    np.random.seed(42)
    X3 = np.random.randn(300, 5)
    y3 = np.random.randint(0, 3, 300)
    print(f"   → {len(X3)} échantillons, {X3.shape[1]} features, 3 classes")
    
    # Test 4: Régression linéaire
    print("\nTest 4: Régression linéaire")
    X4 = np.array([[1], [2], [3], [4], [5]])
    y4 = np.array([2, 4, 6, 8, 10])
    print(f"   → {len(X4)} échantillons, {X4.shape[1]} feature, relation linéaire")
    
    # Sauvegarde
    os.makedirs('tests/data', exist_ok=True)
    save_path = 'tests/data/test_data.npz'
    np.savez(save_path,
             X1=X1, y1=y1, X2=X2, y2=y2, X3=X3, y3=y3, X4=X4, y4=y4)
    
    print(f"\nDonnées sauvegardées dans: {save_path}")
    print("\nGÉNÉRATION TERMINÉE !")
    
    # Afficher un résumé des fichiers
    print("\nContenu du fichier sauvegardé:")
    data = np.load(save_path)
    for key in data.keys():
        print(f"   - {key}: {data[key].shape}")

if __name__ == "__main__":
    generate_simple_test_data()