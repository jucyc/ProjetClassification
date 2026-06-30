"""
Test du MLP (Perceptron Multi-Couches), retropropagation Tanh.

IMPORTANT : ce test appelle NOTRE bibliotheque C via le bridge ctypes
(python_api/ml_bridge.py -> MLPModel). Architecture configurable,
Tanh partout, sorties en -1/+1 (pas de sigmoide/softmax), conformement
au cours (slides "Apprendre - Modele Lineaire et PMC", p.90-95).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from python_api.ml_bridge import MLPModel

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def print_section(title):
    print("\n" + "-"*50)
    print(f"  {title}")
    print("-"*50)


def one_hot(labels, n_classes):
    """Encode des labels entiers en vecteurs one-hot -1.0/+1.0 (pas 0/1)."""
    out = []
    for label in labels:
        row = [-1.0] * n_classes
        row[label] = 1.0
        out.append(row)
    return out


def test_mlp_xor():
    """Test MLP sur XOR avec architecture (2, 2, 1) -- doit reussir,
    contrairement au modele lineaire qui echoue sur ce cas."""
    print_section("Test 1: MLP sur XOR, architecture (2, 2, 1) (doit reussir)")

    X = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    Y = [[-1.0], [1.0], [1.0], [-1.0]]  # sorties en -1/+1

    print("Donnees XOR (non lineairement separables)")
    print("Entrainement (retropropagation Tanh, notre lib C) en cours...")

    model = MLPModel([2, 2, 1])
    model.train(X, Y, learning_rate=0.01, n_iterations=100000)

    correct = 0
    print("\nResultats:")
    for i, x in enumerate(X):
        raw = model.predict_raw(x)[0]
        pred_sign = 1.0 if raw >= 0 else -1.0
        expected = Y[i][0]
        status = "[OK]" if pred_sign == expected else "[FAIL]"
        print(f"   {status} x={x} -> sortie brute={raw:.3f} (signe={pred_sign:+.0f}), attendu={expected:+.0f}")
        if pred_sign == expected:
            correct += 1

    accuracy = correct / len(X)
    print(f"\nPrecision: {accuracy:.2%}")

    if accuracy >= 0.9:
        print("[OK] TEST REUSSI (le MLP reussit la ou le modele lineaire echoue)")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def test_mlp_cross():
    """Test MLP sur un dataset 'croix' (non lineairement separable),
    architecture (2, 4, 1) -- cf. notebook de cas de tests du cours."""
    print_section("Test 2: MLP sur dataset 'Cross', architecture (2, 4, 1)")

    rng = np.random.default_rng(0)
    X = rng.random((500, 2)) * 2.0 - 1.0
    y = np.array([1.0 if (abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3) else -1.0 for p in X])

    print(f"Donnees: {len(X)} echantillons, dataset en croix (non lineaire)")
    print("Entrainement (retropropagation Tanh, notre lib C) en cours...")

    model = MLPModel([2, 4, 1])
    model.train(X.tolist(), [[v] for v in y], learning_rate=0.01, n_iterations=100000)

    preds = np.array([1.0 if model.predict_raw(x)[0] >= 0 else -1.0 for x in X.tolist()])
    accuracy = float(np.mean(preds == y))
    print(f"\nPrecision: {accuracy:.2%}")

    if accuracy >= 0.9:
        print("[OK] TEST REUSSI")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def test_mlp_multiclass():
    """Test MLP multi-classes (3 classes), architecture (2, 3) sans
    couche cachee -- equivalent au cas 'Multi Linear 3 classes' du
    notebook de cas de tests, mais avec un seul MLP (sorties multiples)
    au lieu de 3 perceptrons separes."""
    print_section("Test 3: MLP multi-classes, architecture (2, 3) (doit reussir)")

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
    print("Entrainement (retropropagation Tanh, notre lib C) en cours...")

    model = MLPModel([2, 3])
    Y_onehot = one_hot(y.tolist(), 3)
    model.train(X.tolist(), Y_onehot, learning_rate=0.01, n_iterations=50000)

    preds = np.array([model.predict(x) for x in X.tolist()])
    accuracy = float(np.mean(preds == y))
    print(f"\nPrecision: {accuracy:.2%}")

    if accuracy >= 0.9:
        print("[OK] TEST REUSSI")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def test_mlp_save_load():
    """Verifie que les predictions sont identiques apres save/load."""
    print_section("Test 4: Sauvegarde / chargement du modele")

    X = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    Y = [[-1.0], [1.0], [1.0], [-1.0]]

    model = MLPModel([2, 2, 1])
    model.train(X, Y, learning_rate=0.01, n_iterations=50000)
    preds_before = [model.predict_raw(x)[0] for x in X]

    save_path = os.path.join(os.path.dirname(__file__), "_tmp_mlp_test.bin")
    model.save(save_path)

    model2 = MLPModel([2, 2, 1])
    model2.load(save_path)
    preds_after = [model2.predict_raw(x)[0] for x in X]

    os.remove(save_path)

    identical = np.allclose(preds_before, preds_after)
    print(f"Predictions identiques apres save/load : {identical}")

    if identical:
        print("[OK] TEST REUSSI")
        return True
    else:
        print("[FAIL] TEST ECHOUE")
        return False


def run_all_mlp_tests():
    print("=" * 60)
    print("TESTS DU MLP (Perceptron Multi-Couches, retropropagation Tanh)")
    print("=" * 60)

    results = []

    results.append(("MLP sur XOR (2,2,1)", test_mlp_xor()))
    results.append(("MLP sur Cross (2,4,1)", test_mlp_cross()))
    results.append(("MLP multi-classes (2,3)", test_mlp_multiclass()))
    results.append(("Sauvegarde / chargement", test_mlp_save_load()))

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
        print("\n[OK] TOUS LES TESTS MLP SONT REUSSIS")
    else:
        print("\n[FAIL] CERTAINS TESTS ONT ECHOUE")

    return all_passed


if __name__ == "__main__":
    run_all_mlp_tests()