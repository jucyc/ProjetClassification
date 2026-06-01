import ctypes
import numpy as np
import os
import sys

class LinearModel:
    def __init__(self, n_features=24, n_classes=3):
        # Déterminer l'extension selon l'OS
        if sys.platform == 'win32':
            lib_name = 'libml.dll'
        else:
            lib_name = 'libml.so'
        
        # Chemins possibles (ordre de recherche)
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        lib_paths = [
            os.path.join(base_path, 'lib', lib_name),           # lib/libml.dll
            os.path.join(base_path, 'lib', 'build', lib_name),  # lib/build/libml.dll
            os.path.join(base_path, 'libml', lib_name),         # libml/libml.dll
            os.path.join(base_path, lib_name),                  # ./libml.dll
        ]
        
        self.lib = None
        for path in lib_paths:
            if os.path.exists(path):
                try:
                    self.lib = ctypes.CDLL(path)
                    print(f"Bibliothèque chargée: {path}")
                    break
                except Exception as e:
                    print(f"Erreur chargement {path}: {e}")
        
        if self.lib is None:
            print("\nBIBLIOTHÈQUE NON TROUVÉE !")
            print("   Chemins recherchés:")
            for path in lib_paths:
                print(f"     - {path}")
            print("\n   Solution: Compilez la bibliothèque avec:")
            print("     cd lib")
            print("     gcc -shared -o libml.dll src/linear_model.c src/perceptron.c src/mlp.c -Iinclude -O2")
            raise RuntimeError("Impossible de charger la bibliothèque")
        
        # Définir les types des fonctions
        self.lib.linear_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.linear_create.restype = ctypes.c_void_p
        
        self.lib.linear_train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int
        ]
        
        self.lib.linear_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.linear_predict.restype = ctypes.c_int
        
        self.lib.linear_predict_proba.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.linear_predict_proba.restype = ctypes.POINTER(ctypes.c_float)
        
        self.lib.linear_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.linear_load.argtypes = [ctypes.c_char_p]
        self.lib.linear_load.restype = ctypes.c_void_p
        self.lib.linear_free.argtypes = [ctypes.c_void_p]
        
        # Créer le modèle
        self.obj = self.lib.linear_create(n_features, n_classes)
        self.n_features = n_features
        self.n_classes = n_classes
        print(f"Modèle créé: {n_features} features, {n_classes} classes")
    
    def train(self, X, y, learning_rate=0.01, epochs=1000, verbose=1):
        n_samples = len(X)
        
        # Convertir X en tableau de pointeurs
        X_ptr = (ctypes.POINTER(ctypes.c_float) * n_samples)()
        for i in range(n_samples):
            X_ptr[i] = (ctypes.c_float * len(X[i]))(*X[i])
        
        y_ptr = (ctypes.c_int * n_samples)(*y)
        
        self.lib.linear_train(self.obj, X_ptr, y_ptr, n_samples, 
                              learning_rate, epochs, verbose)
    
    def predict(self, x):
        x_ptr = (ctypes.c_float * len(x))(*x)
        return self.lib.linear_predict(self.obj, x_ptr)
    
    def predict_proba(self, x):
        x_ptr = (ctypes.c_float * len(x))(*x)
        proba_ptr = self.lib.linear_predict_proba(self.obj, x_ptr)
        return [proba_ptr[i] for i in range(self.n_classes)]
    
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