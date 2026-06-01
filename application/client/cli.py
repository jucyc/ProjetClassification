import requests
import base64
import argparse
from PIL import Image
import io

class MLClient:
    def __init__(self, server_url='http://localhost:5000'):
        self.server_url = server_url
    
    def predict_image(self, image_path):
        """Envoie une image au serveur pour prédiction"""
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f'{self.server_url}/predict', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n=== RÉSULTAT DE LA PRÉDICTION ===")
            print(f"Image: {image_path}")
            print(f"Prédiction: {result['prediction']}")
            print(f"Probabilités:")
            for class_name, prob in result['probabilities'].items():
                print(f"  - {class_name}: {prob:.2%}")
            return result
        else:
            print(f"Erreur: {response.status_code}")
            print(response.json())
            return None
    
    def health_check(self):
        response = requests.get(f'{self.server_url}/health')
        return response.json()

def main():
    parser = argparse.ArgumentParser(description='Client de classification d\'images')
    parser.add_argument('image', help='Chemin vers l\'image à classifier')
    parser.add_argument('--server', default='http://localhost:5000', 
                       help='URL du serveur')
    
    args = parser.parse_args()
    
    client = MLClient(args.server)
    
    # Vérifier la santé du serveur
    health = client.health_check()
    print(f"Serveur: {health}")
    
    # Prédire
    client.predict_image(args.image)

if __name__ == '__main__':
    main()