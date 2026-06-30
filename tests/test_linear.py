"""
Test du modele lineaire (Perceptron, regle de Rosenblatt).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from python_api.ml_bridge import LinearModel

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def print_section(title):
    print("\n" + "-"*50)
    print(f"  {title}")
    print("-"*50)


def test_linear_simple():
    """Test sur des données linéairement séparables (doit réussir)"""
    print_section("Test 1: Classification lineaire simple (doit reussir)")

    X = [
        [1, 1], [2, 2], [2, 3], [3, 2],
        [5, 5], [6, 6], [7, 5], [6, 7],
    ]
    y = [0, 0, 0, 0, 1, 1, 1, 1]

    print("Donnees: 8 echantillons, 2 features, 2 classes")
    print("Entrainement (regle de Rosenblatt, notre lib C) en cours...")

    model = LinearModel(n_features=2, n_classes=2)
    model.train(X, y, learning_rate=0.01, n_iterations=20000)

    correct = 0
    print("\nResultats:")
    for i, x in enumerate(X):
        pred = model.predict(x)
        status = "[OK]" if pred == y[i] else "[FAIL]"
        print(f"   {status} x={x} -> predit={pred}, attendu={y[i]}")
        if pred == y[i]:
            correct += 1

    accuracy = correct / len(X)
    print(f"\nPrecision: {accuracy:.2%}")

    if accuracy >= 0.9:
        print("[OK] TEST REUSSI")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def test_linear_xor():
    """Test XOR - doit échouer (non linéairement séparable)"""
    print_section("Test 2: XOR (doit echouer, modele lineaire = limite connue)")

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]

    print("Donnees XOR (non lineairement separables)")
    print("Entrainement (regle de Rosenblatt, notre lib C) en cours...")

    model = LinearModel(n_features=2, n_classes=2)
    model.train(X, y, learning_rate=0.01, n_iterations=20000)

    correct = 0
    print("\nResultats:")
    for i, x in enumerate(X):
        pred = model.predict(x)
        status = "[OK]" if pred == y[i] else "[FAIL]"
        print(f"   {status} x={x} -> predit={pred}, attendu={y[i]}")
        if pred == y[i]:
            correct += 1

    accuracy = correct / len(X)
    print(f"\nPrecision: {accuracy:.2%}")

    # On s'attend a un echec partiel/total : c'est la demonstration
    # pedagogique que le modele lineaire ne peut pas separer XOR.
    if accuracy <= 0.75:
        print("[OK] TEST REUSSI (echec attendu pour un modele lineaire)")
        return True
    else:
        print("[FAIL] Resultat inattendu (le modele a 'reussi' XOR ?)")
        return False


def test_linear_multiclass():
    """Test classification 3 classes (one-vs-rest), doit réussir"""
    print_section("Test 3: Classification 3 classes, one-vs-rest (doit reussir)")

    rng = np.random.default_rng(0)
    X = rng.random((300, 2)) * 2.0 - 1.0
    y = []
    for p in X:
        if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0:
            y.append(0)
        elif -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0:
            y.append(1)
        elif -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0:
            y.append(2)
        else:
            y.append(-1)
    y = np.array(y)
    X = X[y != -1]
    y = y[y != -1]

    print(f"Donnees: {len(X)} echantillons, 2 features, 3 classes")
    print("Entrainement (3 perceptrons one-vs-rest, notre lib C) en cours...")

    model = LinearModel(n_features=2, n_classes=3)
    model.train(X.tolist(), y.tolist(), learning_rate=0.01, n_iterations=30000)

    preds = np.array([model.predict(x) for x in X.tolist()])
    accuracy = float(np.mean(preds == y))
    print(f"\nPrecision: {accuracy:.2%}")

    if accuracy >= 0.9:
        print("[OK] TEST REUSSI")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def run_all_linear_tests():
    print("=" * 60)
    print("TESTS DU MODELE LINEAIRE (Perceptron / Rosenblatt)")
    print("=" * 60)

    results = []

    results.append(("Classification simple", test_linear_simple()))
    results.append(("XOR (doit echouer)", test_linear_xor()))
    results.append(("Multi-classes (one-vs-rest)", test_linear_multiclass()))

    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[OK] TOUS LES TESTS LINEAIRES SONT REUSSIS")
    else:
        print("\n[FAIL] CERTAINS TESTS ONT ECHOUE")

    return all_passed


if __name__ == "__main__":
    run_all_linear_tests()