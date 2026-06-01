import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*50)
print("TEST DE L'EXTRACTION DE FEATURES")
print("="*50)

# Vérifier les imports
print("\n1. Vérification des imports...")
try:
    from PIL import Image
    print("   PIL (Pillow) installé")
except ImportError:
    print("   PIL manquant - pip install pillow")
    sys.exit(1)

try:
    import numpy as np
    print("   NumPy installé")
except ImportError:
    print("   NumPy manquant - pip install numpy")
    sys.exit(1)

# Importer le processeur
from preprocessing.image_processor import ImageProcessor

processor = ImageProcessor()

# Vérifier si des images existent
print("\n2. Recherche des images...")
image_found = False
for class_name in processor.class_names:
    class_dir = f"data/raw/{class_name}"
    if os.path.exists(class_dir):
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            print(f"   {class_name}: {len(files)} images")
            image_found = True
        else:
            print(f"   {class_name}: dossier vide")
    else:
        print(f"   {class_name}: dossier manquant")

if not image_found:
    print("\nAucune image trouvée!")
    print("\nPour ajouter des images :")
    print("   1. Créez les dossiers :")
    print("      mkdir -p data/raw/taj_mahal")
    print("      mkdir -p data/raw/great_wall") 
    print("      mkdir -p data/raw/christ_redeemer")
    print("   2. Ajoutez des images (format .jpg, .png, .jpeg)")
    print("\n   Pour le moment, on peut utiliser les tests simples :")
    print("   python tests/test_linear.py")
else:
    # Tester l'extraction sur la première image trouvée
    print("\n3. Test d'extraction de features...")
    for class_name in processor.class_names:
        class_dir = f"data/raw/{class_name}"
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                test_path = os.path.join(class_dir, files[0])
                print(f"   📷 Test sur: {test_path}")
                try:
                    features = processor.extract_features(test_path)
                    print(f"   Features extraites: {len(features)} dimensions")
                    print(f"    Premières valeurs: {np.array(features[:5]).round(4)}")
                except Exception as e:
                    print(f"   ❌ Erreur: {e}")
                break

print("\n" + "="*50)
print("4. Chargement complet du dataset...")
print("="*50)

X, y = processor.load_dataset('data/raw')

if len(X) > 0:
    print(f"\nDataset chargé avec succès!")
    print(f"   {len(X)} images")
    print(f"   {X.shape[1]} features par image")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Dimensions: {X.shape}")
else:
    print("\n Aucune image chargée")
    print("   Ajoutez des images dans data/raw/taj_mahal/, etc.")

print("\nTest terminé!")
