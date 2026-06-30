from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from python_api.ml_bridge import LinearModel
from preprocessing.image_processor import ImageProcessor

app = Flask(__name__)
CORS(app)

# Variables globales
model = None
processor = ImageProcessor()
class_names = {0: "Taj Mahal", 1: "Grande Muraille", 2: "Christ Rédempteur"}

def load_model():
    global model
    model_path = 'data/models/linear_model.bin'
    if os.path.exists(model_path):
        model = LinearModel()
        model.load(model_path)
        print(f"Modèle chargé: {model_path}")
        return True
    else:
        print("Aucun modèle trouvé. Entraînez d'abord le modèle.")
        return False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400
    
    image_file = request.files['image']
    
    # Sauvegarder temporairement l'image
    temp_path = '/tmp/temp_image.jpg'
    image_file.save(temp_path)
    
    try:
        # Extraire les features
        features = processor.extract_features(temp_path)
        
        # Prédiction
        pred_class = model.predict(list(features))
        scores = model.predict_scores(list(features))
        
        # Nettoyer
        os.remove(temp_path)
        
        return jsonify({
            "prediction": class_names[pred_class],
            "class_id": int(pred_class),
            "scores": {
                class_names[i]: float(scores[i])
                for i in range(3)
            }
        })
    
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    from scripts.train_models import train_linear_model
    
    # Cette fonction serait appelée pour ré-entraîner le modèle
    # À implémenter complètement selon vos besoins
    
    return jsonify({"message": "Fonction d'entraînement à implémenter"})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)