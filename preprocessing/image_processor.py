import numpy as np
import os
from PIL import Image
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size
        self.class_names = ['taj_mahal', 'great_wall', 'christ_redeemer']
        self.class_to_label = {name: i for i, name in enumerate(self.class_names)}
        self.mean = None
        self.std = None
    
    def extract_features(self, image_path):
        """Extrait les pixels bruts de l'image (pour le MLP)"""
        img = Image.open(image_path).convert('L')  # Niveaux de gris
        img = img.resize(self.target_size)
        img_array = np.array(img, dtype=np.float32)
        # Aplatir et normaliser
        features = img_array.flatten() / 255.0
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
    
    def save_processed_data(self, data_dir, output_dir="test_cases", test_size=0.2, random_state=42):
        """Charge, normalise et sauvegarde les données prétraitées"""
        print("="*50)
        print("PRETRAITEMENT DES IMAGES")
        print("="*50)
        print(f"Source: {data_dir}")
        print(f"Sortie: {output_dir}/")
        print(f"Test size: {test_size}")
        print("="*50)
        
        X, y = self.load_dataset(data_dir, normalize=True)
        
        if len(X) == 0:
            print("Aucune image chargée !")
            return None, None, None, None
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        
        print("\n" + "="*50)
        print("SAUVEGARDE")
        print("="*50)
        print(f"X_train.npy: {X_train.shape}")
        print(f"y_train.npy: {y_train.shape}")
        print(f"X_test.npy:  {X_test.shape}")
        print(f"y_test.npy:  {y_test.shape}")
        print(f"\nDonnees sauvegardees dans: {output_dir}/")
        
        return X_train, y_train, X_test, y_test

def main():
    import sys
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data", "raw")
    output_dir = os.path.join(project_dir, "test_cases")
    
    if not os.path.exists(data_dir):
        print(f"Dossier data/raw non trouve: {data_dir}")
        sys.exit(1)
    
    processor = ImageProcessor(target_size=(32, 32))
    X_train, y_train, X_test, y_test = processor.save_processed_data(
        data_dir=data_dir,
        output_dir=output_dir,
        test_size=0.2,
        random_state=42
    )
    
    if X_train is not None:
        print("\nPretraitement termine avec succes !")
        print(f"   Train: {X_train.shape[0]} images")
        print(f"   Test:  {X_test.shape[0]} images")
        print(f"   Features: {X_train.shape[1]}")
    else:
        print("\nLe pretraitement a echoue.")

if __name__ == "__main__":
    main()
