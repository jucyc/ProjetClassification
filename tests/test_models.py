import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_api.ml_bridge import LinearModel

def test_linear_classification():
    print("\n=== Test Classification Linéaire ===")
    
    # Charger les données de test
    data = np.load('tests/data/test_data.npz')
    X = data['X1']
    y = data['y1']
    
    # Créer et entraîner le modèle
    model = LinearModel(n_features=X.shape[1], n_classes=2)
    X_list = [list(x) for x in X]
    y_list = list(y)
    
    model.train(X_list, y_list, learning_rate=0.01, epochs=500)
    
    # Évaluation
    correct = 0
    for i, x in enumerate(X_list):
        pred = model.predict(x)
        if pred == y_list[i]:
            correct += 1
        print(f"Entrée: {x}, Prédit: {pred}, Attendu: {y_list[i]}")
    
    accuracy = correct / len(X)
    print(f"Précision: {accuracy:.2%}")
    return accuracy == 1.0

def test_xor():
    print("\n=== Test XOR (non linéaire) ===")
    
    data = np.load('tests/data/test_data.npz')
    X = data['X2']
    y = data['y2']
    
    # Pour XOR, le modèle linéaire ne doit PAS réussir
    model = LinearModel(n_features=2, n_classes=2)
    X_list = [list(x) for x in X]
    y_list = list(y)
    
    model.train(X_list, y_list, learning_rate=0.01, epochs=500)
    
    correct = 0
    for i, x in enumerate(X_list):
        pred = model.predict(x)
        if pred == y_list[i]:
            correct += 1
        print(f"Entrée: {x}, Prédit: {pred}, Attendu: {y_list[i]}")
    
    accuracy = correct / len(X)
    print(f"Précision: {accuracy:.2%}")
    
    return accuracy <= 0.75

def test_image_features():
    print("\n=== Test Extraction de Features d'Images ===")
    
    from preprocessing.image_processor import ImageProcessor
    
    processor = ImageProcessor()
    

    test_image_path = "data/raw/taj_mahal/sample.jpg"  
    
    if os.path.exists(test_image_path):
        features = processor.extract_combined_features(test_image_path)
        print(f"Dimensions des features: {len(features)}")
        print(f"Premières features: {features[:10]}")
        return True
    else:
        print(f"Image de test non trouvée: {test_image_path}")
        return False

if __name__ == "__main__":
    print("=== VALIDATION DES MODÈLES ===")
    
    success = True
    
    # Test 1: Classification linéaire (doit réussir)
    if test_linear_classification():
        print("Classification linéaire: OK")
    else:
        print(" lassification linéaire: KO")
        success = False
    
    # Test 2: XOR (doit échouer pour le modèle linéaire)
    if test_xor():
        print("XOR (modèle linéaire): OK (normalement échoue)")
    else:
        print("XOR (modèle linéaire): KO (aurait dû échouer)")
    
    # Test 3: Extraction de features
    test_image_features()
    
    if success:
        print("\nTOUS LES TESTS RÉUSSIS")
    else:
        print("\nCERTAINS TESTS ONT ÉCHOUÉ")