"""
Test du MLP (Perceptron Multi-Couches)
"""

import sys
import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

import numpy as np
import ctypes

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def print_section(title):
    print("\n" + "-"*50)
    print(f"  {title}")
    print("-"*50)

def test_mlp_xor():
    """Test MLP sur XOR"""
    print_section("Test MLP sur XOR")
    
    possible_paths = [
        os.path.join(project_dir, "lib", "mlp.so"),
        os.path.join(project_dir, "lib", "mlp.dll"),
        "../lib/mlp.so",
        "../lib/mlp.dll",
    ]
    
    lib_path = None
    for path in possible_paths:
        if os.path.exists(path):
            lib_path = path
            break
    
    if lib_path is None:
        print("[FAIL] Erreur: librairie non trouvee")
        return False
    
    print(f"Librairie chargee: {lib_path}")
    lib = ctypes.CDLL(lib_path)
    
    lib.mlp_create.restype = ctypes.c_void_p
    lib.mlp_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mlp_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                              ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                              ctypes.c_double, ctypes.c_int, ctypes.c_int]
    lib.mlp_predict.restype = ctypes.c_int
    lib.mlp_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.mlp_evaluate.restype = ctypes.c_double
    lib.mlp_evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                 ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.mlp_destroy.argtypes = [ctypes.c_void_p]
    
    def to_c_double(arr):
        return arr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    def to_c_int(arr):
        return arr.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    # Donnees XOR
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    
    print("Donnees XOR (probleme non lineaire)")
    print("Entrainement du MLP...")
    
    model = lib.mlp_create(2, 4, 2)
    
    n_samples = ctypes.c_int(4)
    
    lib.mlp_train(model, to_c_double(X), to_c_int(y), n_samples, ctypes.c_double(0.8), ctypes.c_int(1000), ctypes.c_int(4))
    
    acc = lib.mlp_evaluate(model, to_c_double(X), to_c_int(y), n_samples)
    print(f"Precision: {acc:.1f}%")
    
    print("\nResultats:")
    for i in range(4):
        # CORRECTION : Créer un tableau pour chaque échantillon
        x_sample = np.array([X[i]], dtype=np.float64)
        pred = lib.mlp_predict(model, to_c_double(x_sample))
        status = "[OK]" if pred == y[i] else "[FAIL]"
        print(f"   {status} x={X[i]} -> predit={pred}, attendu={y[i]}")
    
    lib.mlp_destroy(model)
    
    if acc >= 80.0:
        print("\n[OK] TEST REUSSI")
        return True
    else:
        print("\n[FAIL] TEST ECHOUE (accuracy < 80%)")
        return False

def test_mlp_monuments():
    """Test MLP sur dataset monuments"""
    print_section("Test MLP sur Monuments")
    
    try:
        data_paths = [
            os.path.join(project_dir, "test_cases", "X_train.npy"),
            os.path.join(project_dir, "data", "processed", "X_train.npy"),
            "../test_cases/X_train.npy",
        ]
        
        X_train = None
        y_train = None
        for path in data_paths:
            if os.path.exists(path):
                X_train = np.load(path)
                y_path = path.replace("X_train", "y_train")
                if os.path.exists(y_path):
                    y_train = np.load(y_path)
                break
        
        if X_train is None or y_train is None:
            print("[FAIL] Donnees non trouvees")
            return False
        
        print(f"Donnees: {len(X_train)} images")
        
        lib_path = os.path.join(project_dir, "lib", "mlp.so")
        if not os.path.exists(lib_path):
            lib_path = os.path.join(project_dir, "lib", "mlp.dll")
        
        lib = ctypes.CDLL(lib_path)
        
        lib.mlp_create.restype = ctypes.c_void_p
        lib.mlp_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.mlp_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                  ctypes.c_double, ctypes.c_int, ctypes.c_int]
        lib.mlp_evaluate.restype = ctypes.c_double
        lib.mlp_evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.mlp_destroy.argtypes = [ctypes.c_void_p]
        
        def to_c_double(arr):
            return arr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        def to_c_int(arr):
            return arr.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        print("Entrainement sur monuments...")
        model = lib.mlp_create(1024, 64, 3)
        
        n_samples = ctypes.c_int(len(X_train))
        lib.mlp_train(model, to_c_double(X_train), to_c_int(y_train),
                      n_samples, ctypes.c_double(0.01), ctypes.c_int(200), ctypes.c_int(32))
        
        train_acc = lib.mlp_evaluate(model, to_c_double(X_train), to_c_int(y_train), n_samples)
        print(f"Train Accuracy: {train_acc:.2f}%")
        
        lib.mlp_destroy(model)
        
        if train_acc > 50:
            print("[OK] TEST REUSSI")
            return True
        else:
            print("[FAIL] TEST ECHOUE")
            return False
        
    except Exception as e:
        print(f"Erreur: {e}")
        return False

def run_all_mlp_tests():
    print("="*60)
    print("TESTS DU MLP")
    print("="*60)
    
    results = []
    
    results.append(("MLP sur XOR", test_mlp_xor()))
    results.append(("MLP sur Monuments", test_mlp_monuments()))
    
    print("\n" + "="*60)
    print("RESUME")
    print("="*60)
    
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
