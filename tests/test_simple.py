"""Test simple pour vérifier que tout fonctionne"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from python_api.ml_bridge import LinearModel

def test_basic():
    print("="*60)
    print("TEST SIMPLE - VÉRIFICATION DE BASE")
    print("="*60)
    
    # Données simples : AND logique
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]  # AND
    
    print("\n Données d'entraînement:")
    for i, x in enumerate(X):
        print(f"   {x} → {y[i]}")
    
    print("\nEntraînement du modèle...")
    model = LinearModel(n_features=2, n_classes=2)
    model.train(X, y, learning_rate=0.1, epochs=500, verbose=1)
    
    print("\nRésultats des prédictions:")
    correct = 0
    for i, x in enumerate(X):
        pred = model.predict(x)
        status = "ok" if pred == y[i] else "ko"
        print(f"   {status} Entrée: {x} → Prédit: {pred}, Attendu: {y[i]}")
        if pred == y[i]:
            correct += 1
    
    accuracy = correct / len(X)
    print(f"\n Précision: {accuracy:.2%}")
    
    if accuracy == 1.0:
        print("\n TEST RÉUSSI !")
    else:
        print("\n TEST ÉCHOUÉ")
    
    return accuracy == 1.0

if __name__ == "__main__":
    test_basic()