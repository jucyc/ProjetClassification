"""Test simple pour vérifier le chargement de la bibliothèque"""
import os
import sys
import ctypes

print("="*50)
print("TEST DE CHARGEMENT DE LA BIBLIOTHÈQUE")
print("="*50)

# Vérifier les fichiers présents
print("\nFichiers dans le dossier lib:")
lib_dir = "lib"
if os.path.exists(lib_dir):
    for f in os.listdir(lib_dir):
        if f.endswith('.dll') or f.endswith('.so'):
            size = os.path.getsize(os.path.join(lib_dir, f))
            print(f"   - {f} ({size} bytes)")
else:
    print(f"   Dossier {lib_dir} non trouvé")

# Tester l'import du module
print("\nTest d'import du modèle...")
try:
    from python_api.ml_bridge import LinearModel
    print("Module importé avec succès")
except Exception as e:
    print(f"Erreur d'import: {e}")
    sys.exit(1)

# Tester la création du modèle
print("\nTest de création du modèle...")
try:
    model = LinearModel(n_features=2, n_classes=2)
    print("Modèle créé avec succès")
except Exception as e:
    print(f"Erreur de création: {e}")
    sys.exit(1)

print("\nTOUS LES TESTS RÉUSSIS !")