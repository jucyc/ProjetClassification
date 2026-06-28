"""
Tests du RBF Network
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from python_api.rbf_bridge import RBF


def labels_binaire(y):
    return [1 if v == 1 else 0 for v in y]


def labels_onehot_vers_index(Y):
    return [int(np.argmax(row)) for row in Y]


def print_section(title, attendu):
    print("\n" + "-" * 60)
    print(f"  {title}   (attendu d'apres le cours : {attendu})")
    print("-" * 60)


def run_case(X, y, n_classes, n_centers, gamma, lr, epochs, name, attendu):
    print_section(name, attendu)
    X = X.tolist() if isinstance(X, np.ndarray) else X

    n_centers = min(n_centers, len(X))
    model = RBF(input_size=len(X[0]), n_centers=n_centers, output_size=n_classes, gamma=gamma)
    model.fit_centers(X, kmeans_iters=20)
    model.train(X, y, learning_rate=lr, epochs=epochs)

    acc = model.evaluate(X, y)
    print(f"\nPrecision finale : {acc:.2f}%")
    return acc


def test_linear_simple():
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])
    Y = np.array([1, -1, -1])
    y = labels_binaire(Y)
    acc = run_case(X, y, n_classes=2, n_centers=3, gamma=0.5, lr=0.3, epochs=300,
                    name="Linear Simple", attendu="OK")
    assert acc >= 99.0


def test_linear_multiple():
    X = np.concatenate([
        np.random.random((50, 2)) * 0.9 + np.array([1, 1]),
        np.random.random((50, 2)) * 0.9 + np.array([2, 2])
    ])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0]).flatten()
    y = labels_binaire(Y)
    acc = run_case(X, y, n_classes=2, n_centers=10, gamma=1.0, lr=0.3, epochs=300,
                    name="Linear Multiple", attendu="OK")
    assert acc >= 90.0


def test_xor():
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])
    y = labels_binaire(Y)
    acc = run_case(X, y, n_classes=2, n_centers=4, gamma=2.0, lr=0.5, epochs=1000,
                    name="XOR", attendu="KO pour lineaire, OK pour MLP -> on attend OK pour RBF")
    assert acc >= 99.0


def test_cross():
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])
    y = labels_binaire(Y)
    acc = run_case(X, y, n_classes=2, n_centers=30, gamma=3.0, lr=0.3, epochs=500,
                    name="Cross", attendu="KO pour lineaire, OK pour MLP -> on attend OK pour RBF")
    assert acc >= 85.0


def test_multi_linear_3_classes():
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [-1, -1, -1] for p in X])

    mask = [not np.all(arr == [-1, -1, -1]) for arr in Y]
    X = X[mask]
    Y = Y[mask]

    y = labels_onehot_vers_index(Y)
    acc = run_case(X, y, n_classes=3, n_centers=20, gamma=2.0, lr=0.3, epochs=500,
                    name="Multi Linear 3 classes", attendu="OK")
    assert acc >= 90.0


def test_multi_cross():
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else
                  [-1, 1, -1] if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else
                  [-1, -1, 1] for p in X])

    y = labels_onehot_vers_index(Y)
    acc = run_case(X, y, n_classes=3, n_centers=150, gamma=20.0, lr=0.5, epochs=1500,
                    name="Multi Cross", attendu="KO pour lineaire, OK pour MLP -> on attend OK pour RBF")
    assert acc >= 80.0


if __name__ == "__main__":
    print("=" * 60)
    print("  TESTS DU RBF NETWORK")
    print("=" * 60)

    tests = [
        test_linear_simple,
        test_linear_multiple,
        test_xor,
        test_cross,
        test_multi_linear_3_classes,
        test_multi_cross,
    ]
    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print("=> OK\n")
        except AssertionError:
            print("=> ECHEC : precision insuffisante\n")
        except Exception as e:
            print(f"=> ERREUR : {e}\n")

    print("=" * 60)
    print(f"  Resultat : {passed}/{len(tests)} tests reussis")
    print("=" * 60)