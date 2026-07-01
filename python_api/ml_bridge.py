import ctypes
import numpy as np
import os
import sys

def _load_lib():
    if sys.platform == 'win32':
        lib_name = 'libml.dll'
    else:
        lib_name = 'libml.so'

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    lib_paths = [
        os.path.join(base_path, 'lib', lib_name),
        os.path.join(base_path, 'lib', 'build', lib_name),
        os.path.join(base_path, 'libml', lib_name),
        os.path.join(base_path, lib_name),
    ]

    for path in lib_paths:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                print(f"Bibliothèque chargée: {path}")
                return lib
            except Exception as e:
                print(f"Erreur chargement {path}: {e}")

    print("\nBIBLIOTHÈQUE NON TROUVÉE !")
    print("   Chemins recherchés:")
    for path in lib_paths:
        print(f"     - {path}")
    print("\n   Solution: Compilez la bibliothèque avec:")
    print("     cd lib")
    print("     gcc -shared -fPIC -o libml.so src/linear_model.c src/mlp.c -Iinclude -lm -O2")
    raise RuntimeError("Impossible de charger la bibliothèque")


class LinearModel:
    def __init__(self, n_features=24, n_classes=3):
        self.lib = _load_lib()
        
        self.lib.linear_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.linear_create.restype = ctypes.c_void_p
        
        self.lib.linear_train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
        ]
        
        self.lib.linear_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.linear_predict.restype = ctypes.c_int
        
        self.lib.linear_predict_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.linear_predict_scores.restype = ctypes.POINTER(ctypes.c_float)
        
        self.lib.linear_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.linear_load.argtypes = [ctypes.c_char_p]
        self.lib.linear_load.restype = ctypes.c_void_p
        self.lib.linear_free.argtypes = [ctypes.c_void_p]
        
        self.obj = self.lib.linear_create(n_features, n_classes)
        self.n_features = n_features
        self.n_classes = n_classes
        print(f"Modèle créé: {n_features} features, {n_classes} classes")
    
    def train(self, X, y, learning_rate=0.02, n_iterations=50000):
        """ 
        n_iterations = nombre de tirages aleatoires d'exemples PAR CLASSE
        """
        n_samples = len(X)
        
        X_ptr = (ctypes.POINTER(ctypes.c_float) * n_samples)()
        for i in range(n_samples):
            X_ptr[i] = (ctypes.c_float * len(X[i]))(*X[i])
        
        y_ptr = (ctypes.c_int * n_samples)(*y)
        
        self.lib.linear_train(self.obj, X_ptr, y_ptr, n_samples, 
                              learning_rate, n_iterations)
    
    def predict(self, x):
        x_ptr = (ctypes.c_float * len(x))(*x)
        return self.lib.linear_predict(self.obj, x_ptr)
    
    def predict_scores(self, x):
        x_ptr = (ctypes.c_float * len(x))(*x)
        scores_ptr = self.lib.linear_predict_scores(self.obj, x_ptr)
        return [scores_ptr[i] for i in range(self.n_classes)]
    
    def save(self, filename):
        self.lib.linear_save(self.obj, filename.encode('utf-8'))
        print(f"Modèle sauvegardé: {filename}")
    
    def load(self, filename):
        if self.obj:
            self.lib.linear_free(self.obj)
        self.obj = self.lib.linear_load(filename.encode('utf-8'))
        print(f"Modèle chargé: {filename}")
    
    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            self.lib.linear_free(self.obj)


class MLPModel:

    def __init__(self, npl):
        self.lib = _load_lib()

        self.lib.mlp_create.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.mlp_create.restype = ctypes.c_void_p

        self.lib.mlp_train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
        ]

        self.lib.mlp_predict_raw.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.mlp_predict_raw.restype = ctypes.POINTER(ctypes.c_double)

        self.lib.mlp_predict_class.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.mlp_predict_class.restype = ctypes.c_int

        self.lib.mlp_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.mlp_load.argtypes = [ctypes.c_char_p]
        self.lib.mlp_load.restype = ctypes.c_void_p
        self.lib.mlp_free.argtypes = [ctypes.c_void_p]

        self.npl = list(npl)
        npl_arr = (ctypes.c_int * len(npl))(*npl)
        self.obj = self.lib.mlp_create(npl_arr, len(npl))
        self.n_outputs = npl[-1]
        print(f"MLP créé: architecture {self.npl}")

    def train(self, X, Y, learning_rate=0.08, n_iterations=30):
        n_samples = len(X)

        X_ptr = (ctypes.POINTER(ctypes.c_double) * n_samples)()
        for i in range(n_samples):
            X_ptr[i] = (ctypes.c_double * len(X[i]))(*[float(v) for v in X[i]])

        Y_ptr = (ctypes.POINTER(ctypes.c_double) * n_samples)()
        for i in range(n_samples):
            Y_ptr[i] = (ctypes.c_double * len(Y[i]))(*[float(v) for v in Y[i]])

        self.lib.mlp_train(self.obj, X_ptr, Y_ptr, n_samples, learning_rate, n_iterations)

    def predict_raw(self, x):
        """Sorties brutes en -1/+1"""
        x_ptr = (ctypes.c_double * len(x))(*[float(v) for v in x])
        out_ptr = self.lib.mlp_predict_raw(self.obj, x_ptr)
        return [out_ptr[i] for i in range(self.n_outputs)]

    def predict(self, x):
        """Classe predite"""
        x_ptr = (ctypes.c_double * len(x))(*[float(v) for v in x])
        return self.lib.mlp_predict_class(self.obj, x_ptr)

    def save(self, filename):
        self.lib.mlp_save(self.obj, filename.encode('utf-8'))
        print(f"Modèle sauvegardé: {filename}")

    def load(self, filename):
        if self.obj:
            self.lib.mlp_free(self.obj)
        self.obj = self.lib.mlp_load(filename.encode('utf-8'))
        print(f"Modèle chargé: {filename}")

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            self.lib.mlp_free(self.obj)