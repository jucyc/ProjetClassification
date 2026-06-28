import ctypes
import os
import sys


class RBF:
    def __init__(self, input_size, n_centers, output_size, gamma=0.1):
        if sys.platform == 'win32':
            lib_name = 'libml.dll'
        else:
            lib_name = 'libml.so'

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lib_paths = [
            os.path.join(base_path, 'lib', lib_name),
            os.path.join(base_path, 'lib', 'build', lib_name),
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
            raise RuntimeError("Impossible de charger la bibliothèque")

        # Définition des types de fonctions C
        self.lib.rbf_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self.lib.rbf_create.restype = ctypes.c_void_p

        self.lib.rbf_fit_centers.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
        ]

        self.lib.rbf_train.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int
        ]

        self.lib.rbf_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.rbf_predict.restype = ctypes.c_int

        self.lib.rbf_evaluate.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int
        ]
        self.lib.rbf_evaluate.restype = ctypes.c_double

        self.lib.rbf_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.rbf_load.argtypes = [ctypes.c_char_p]
        self.lib.rbf_load.restype = ctypes.c_void_p
        self.lib.rbf_destroy.argtypes = [ctypes.c_void_p]

        self.input_size = input_size
        self.n_centers = n_centers
        self.output_size = output_size
        self.obj = self.lib.rbf_create(input_size, n_centers, output_size, gamma)
        print(f"RBF créé: input={input_size}, centres={n_centers}, output={output_size}, gamma={gamma}")

    @staticmethod
    def _flatten(X):
        """Convertit une liste de listes en tableau ctypes 1D (comme attendu par le C)."""
        flat = [v for row in X for v in row]
        return (ctypes.c_double * len(flat))(*flat)

    def fit_centers(self, X, kmeans_iters=20):
        n_samples = len(X)
        X_flat = self._flatten(X)
        self.lib.rbf_fit_centers(self.obj, X_flat, n_samples, kmeans_iters)

    def train(self, X, y, learning_rate=0.1, epochs=500, batch_size=None):
        n_samples = len(X)
        if batch_size is None:
            batch_size = n_samples
        X_flat = self._flatten(X)
        y_arr = (ctypes.c_int * n_samples)(*y)
        self.lib.rbf_train(self.obj, X_flat, y_arr, n_samples, learning_rate, epochs, batch_size)

    def predict(self, x):
        x_arr = (ctypes.c_double * len(x))(*x)
        return self.lib.rbf_predict(self.obj, x_arr)

    def predict_batch(self, X):
        return [self.predict(x) for x in X]

    def evaluate(self, X, y):
        n_samples = len(X)
        X_flat = self._flatten(X)
        y_arr = (ctypes.c_int * n_samples)(*y)
        return self.lib.rbf_evaluate(self.obj, X_flat, y_arr, n_samples)

    def save(self, filename):
        self.lib.rbf_save(self.obj, filename.encode('utf-8'))

    def load(self, filename):
        if self.obj:
            self.lib.rbf_destroy(self.obj)
        self.obj = self.lib.rbf_load(filename.encode('utf-8'))

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            self.lib.rbf_destroy(self.obj)