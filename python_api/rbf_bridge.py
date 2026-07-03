import ctypes
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_bridge import _load_lib


class RBFModel:
    """
    Reseau RBF (Radial Basis Function)
    """

    def __init__(self, n_centers=20, n_features=1024, n_classes=3, gamma=0.001):
        self.lib = _load_lib()

        self.lib.rbf_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double
        ]
        self.lib.rbf_create.restype = ctypes.c_void_p

        self.lib.rbf_train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.lib.rbf_predict.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)
        ]
        self.lib.rbf_predict.restype = ctypes.c_int

        self.lib.rbf_predict_scores.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)
        ]
        self.lib.rbf_predict_scores.restype = ctypes.POINTER(ctypes.c_double)

        self.lib.rbf_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.rbf_load.argtypes = [ctypes.c_char_p]
        self.lib.rbf_load.restype = ctypes.c_void_p
        self.lib.rbf_free.argtypes = [ctypes.c_void_p]

        self.n_centers  = n_centers
        self.n_features = n_features
        self.n_classes  = n_classes
        self.gamma      = gamma

        self.obj = self.lib.rbf_create(n_centers, n_features, n_classes, gamma)
        print(f"RBF créé: {n_centers} centres, gamma={gamma}, {n_classes} classes")

    def train(self, X, y, n_iter=100):
        n_samples = len(X)
        X_ptr = (ctypes.POINTER(ctypes.c_double) * n_samples)()
        for i in range(n_samples):
            X_ptr[i] = (ctypes.c_double * len(X[i]))(*[float(v) for v in X[i]])
        y_ptr = (ctypes.c_int * n_samples)(*[int(v) for v in y])
        self.lib.rbf_train(self.obj, X_ptr, y_ptr, n_samples, n_iter)

    def predict(self, x):
        x_ptr = (ctypes.c_double * len(x))(*[float(v) for v in x])
        return self.lib.rbf_predict(self.obj, x_ptr)

    def predict_scores(self, x):
        x_ptr = (ctypes.c_double * len(x))(*[float(v) for v in x])
        scores_ptr = self.lib.rbf_predict_scores(self.obj, x_ptr)
        return [scores_ptr[i] for i in range(self.n_classes)]

    def save(self, filename):
        self.lib.rbf_save(self.obj, filename.encode('utf-8'))
        print(f"Modèle RBF sauvegardé: {filename}")

    def load(self, filename):
        if self.obj:
            self.lib.rbf_free(self.obj)
        self.obj = self.lib.rbf_load(filename.encode('utf-8'))
        print(f"Modèle RBF chargé: {filename}")

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            self.lib.rbf_free(self.obj)