import numpy as np
import os
from PIL import Image
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
        self.class_names = ['taj_mahal', 'great_wall', 'christ_redeemer']
        self.class_to_label = {name: i for i, name in enumerate(self.class_names)}
        self.mean = None
        self.std = None
    
    def extract_features(self, image_path):
        """Extrait des caractéristiques d'une image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.target_size)
        img_array = np.array(img, dtype=np.float32)
        
        # 1. Histogramme de couleurs
        hist_features = []
        for channel in range(3):
            hist, _ = np.histogram(img_array[:, :, channel].flatten(), bins=16, range=(0, 256))
            hist = hist / (np.sum(hist) + 1e-7)
            hist_features.extend(hist)
        
        # 2. Statistiques de base
        gray = np.mean(img_array, axis=2)
        mean_val = np.mean(gray) / 255.0
        std_val = np.std(gray) / 255.0
        
        # 3. Gradients
        gy, gx = np.gradient(gray)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        grad_mean = np.mean(gradient_magnitude) / 255.0
        grad_std = np.std(gradient_magnitude) / 255.0
        
        features = np.concatenate([hist_features, [mean_val, std_val, grad_mean, grad_std]])
        return features
    
    def normalize(self, X, fit=True):
        """Normalise les features (moyenne=0, écart-type=1)"""
        X = np.array(X)
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            self.std[self.std == 0] = 1.0
        
        X_normalized = (X - self.mean) / (self.std + 1e-8)
        return X_normalized
    
    def load_dataset(self, data_dir, normalize=True):
        """Charge et normalise le dataset"""
        X = []
        y = []
        
        print("="*50)
        print("CHARGEMENT DU DATASET")
        print("="*50)
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Dossier non trouvé: {class_dir}")
                continue
            
            print(f"\nChargement de {class_name}...")
            files = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for filename in tqdm(files, desc=f"   {class_name}"):
                image_path = os.path.join(class_dir, filename)
                try:
                    features = self.extract_features(image_path)
                    X.append(features)
                    y.append(self.class_to_label[class_name])
                except Exception as e:
                    print(f"   Erreur avec {filename}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        if normalize and len(X) > 0:
            X = self.normalize(X, fit=True)
        
        print(f"\nDataset chargé: {len(X)} images, features: {X.shape[1]}")
        print(f"   Dimensions: {X.shape}")
        
        return X, y