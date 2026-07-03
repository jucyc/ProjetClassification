"""
Test du modele RBF (Radial Basis Function).
K-means (Lloyd) pour les centres + pseudo-inverse pour les poids.
Conforme aux slides "RBF, SVM et Conclusion", p.99-112.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from python_api.rbf_bridge import RBFModel

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def print_section(title):
    print("\n" + "-"*50)
    print(f"  {title}")
    print("-"*50)


def test_rbf_linear():
    """Cas lineairement separable (doit reussir)"""
    print_section("Test 1: Cas lineaire simple (doit reussir)")

    X = [[1.0,1.0],[2.0,2.0],[2.0,3.0],[3.0,2.0],
         [5.0,5.0],[6.0,6.0],[7.0,5.0],[6.0,7.0]]
    y = [0,0,0,0,1,1,1,1]

    model = RBFModel(n_centers=4, n_features=2, n_classes=2, gamma=1.0)
    model.train(X, y, n_iter=50)

    preds = [model.predict(x) for x in X]
    acc = sum(p==t for p,t in zip(preds,y)) / len(y)
    print(f"Precision: {acc:.2%}")

    if acc >= 0.9:
        print("[OK] TEST REUSSI")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def test_rbf_xor():
    """XOR (doit reussir avec RBF, contrairement au modele lineaire)"""
    print_section("Test 2: XOR (doit reussir avec RBF)")

    X = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
    y = [0,1,1,0]

    model = RBFModel(n_centers=4, n_features=2, n_classes=2, gamma=1.0)
    model.train(X, y, n_iter=50)

    preds = [model.predict(x) for x in X]
    acc = sum(p==t for p,t in zip(preds,y)) / len(y)
    print(f"Precision: {acc:.2%}")

    if acc >= 0.9:
        print("[OK] TEST REUSSI (RBF resout XOR, contrairement au modele lineaire)")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def test_rbf_multiclass():
    """Classification 3 classes (doit reussir)"""
    print_section("Test 3: Multi-classes 3 classes (doit reussir)")

    rng = np.random.default_rng(0)
    X = rng.random((300, 2)) * 2.0 - 1.0
    y = []
    for p in X:
        if -p[0]-p[1]-0.5>0 and p[1]<0 and p[0]-p[1]-0.5<0: y.append(0)
        elif -p[0]-p[1]-0.5<0 and p[1]>0 and p[0]-p[1]-0.5<0: y.append(1)
        elif -p[0]-p[1]-0.5<0 and p[1]<0 and p[0]-p[1]-0.5>0: y.append(2)
        else: y.append(-1)
    y = np.array(y)
    X = X[y != -1].tolist()
    y = y[y != -1].tolist()

    print(f"Donnees: {len(X)} echantillons, 3 classes")

    model = RBFModel(n_centers=10, n_features=2, n_classes=3, gamma=2.0)
    model.train(X, y, n_iter=100)

    preds = [model.predict(x) for x in X]
    acc = sum(p==t for p,t in zip(preds,y)) / len(y)
    print(f"Precision: {acc:.2%}")

    if acc >= 0.9:
        print("[OK] TEST REUSSI")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def test_rbf_save_load():
    """Sauvegarde et chargement"""
    print_section("Test 4: Sauvegarde / chargement")

    X = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
    y = [0,1,1,0]

    model = RBFModel(n_centers=4, n_features=2, n_classes=2, gamma=1.0)
    model.train(X, y, n_iter=50)
    preds_before = [model.predict(x) for x in X]

    save_path = os.path.join(os.path.dirname(__file__), "_tmp_rbf_test.bin")
    model.save(save_path)

    model2 = RBFModel(n_centers=4, n_features=2, n_classes=2, gamma=1.0)
    model2.load(save_path)
    preds_after = [model2.predict(x) for x in X]

    os.remove(save_path)

    identical = preds_before == preds_after
    print(f"Predictions identiques apres save/load: {identical}")

    if identical:
        print("[OK] TEST REUSSI")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def run_all_rbf_tests():
    print("=" * 60)
    print("TESTS DU RBF (Radial Basis Function)")
    print("=" * 60)

    results = []
    results.append(("Cas lineaire simple", test_rbf_linear()))
    results.append(("XOR (doit reussir)", test_rbf_xor()))
    results.append(("Multi-classes (3 classes)", test_rbf_multiclass()))
    results.append(("Sauvegarde / chargement", test_rbf_save_load()))

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
        print("\n[OK] TOUS LES TESTS RBF SONT REUSSIS")
    else:
        print("\n[FAIL] CERTAINS TESTS ONT ECHOUE")

    return all_passed


if __name__ == "__main__":
    run_all_rbf_tests()