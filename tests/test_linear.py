import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from python_api.ml_bridge import LinearModel

def print_section(title):
    print("\n" + "-"*50)
    print(f"  {title}")
    print("-"*50)

def test_linear_simple():
    """Test sur des données linéairement séparables"""
    print_section("Test 1: Classification linéaire simple")
    
    X = [
        [1, 1], [2, 2], [2, 3], [3, 2],  # Classe 0
        [5, 5], [6, 6], [7, 5], [6, 7],  # Classe 1
    ]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    
    print("Données: 8 échantillons, 2 features, 2 classes")
    print("Entraînement en cours...")
    
    model = LinearModel(n_features=2, n_classes=2)
    model.train(X, y, learning_rate=0.01, epochs=500, verbose=0)
    
    correct = 0
    print("\nRésultats:")
    for i, x in enumerate(X):
        pred = model.predict(x)
        status = "✓" if pred == y[i] else "✗"
        print(f"   {status} x={x} → prédit={pred}, attendu={y[i]}")
        if pred == y[i]:
            correct += 1
    
    accuracy = correct / len(X)
    print(f"\nPrécision: {accuracy:.2%}")
    
    if accuracy == 1.0:
        print("TEST RÉUSSI")
        return True
    else:
        print("TEST ÉCHOUÉ")
        return False

def test_linear_xor():
    """Test XOR - doit échouer"""
    print_section("Test 2: XOR (doit échouer pour modèle linéaire)")
    
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]  # XOR
    
    print("Données XOR (non linéairement séparables)")
    print("Entraînement en cours...")
    
    model = LinearModel(n_features=2, n_classes=2)
    model.train(X, y, learning_rate=0.01, epochs=1000, verbose=0)
    
    correct = 0
    print("\nRésultats:")
    for i, x in enumerate(X):
        pred = model.predict(x)
        status = "✓" if pred == y[i] else "✗"
        print(f"   {status} x={x} → prédit={pred}, attendu={y[i]}")
        if pred == y[i]:
            correct += 1
    
    accuracy = correct / len(X)
    print(f"\nPrécision: {accuracy:.2%}")
    
    # Le modèle linéaire ne peut pas résoudre XOR
    # On s'attend à une précision <= 75% (3/4 ou moins)
    if accuracy <= 0.75:
        print("TEST RÉUSSI (échec normal pour modèle linéaire)")
        return True
    else:
        print("Résultat inattendu (le modèle a réussi XOR?)")
        return False

def run_all_linear_tests():
    print("="*60)
    print("TESTS DU MODÈLE LINÉAIRE")
    print("="*60)
    
    results = []
    
    # Test 1
    results.append(("Classification simple", test_linear_simple()))
    
    # Test 2
    results.append(("XOR (doit échouer)", test_linear_xor()))
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    
    for name, passed in results:
        status = "ok" if passed else "ko"
        print(f"  {status} {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\nTOUS LES TESTS LINÉAIRES SONT RÉUSSIS")
    else:
        print("\nCERTAINS TESTS ONT ÉCHOUÉ")
    
    return all_passed

if __name__ == "__main__":
    run_all_linear_tests()