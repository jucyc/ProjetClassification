import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.image_processor import ImageProcessor

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== CHARGEMENT DU MODELE (une seule fois au démarrage) ====================

MODEL_PATH = 'data/models/linear_model_pure.npz'
STATS_PATH = 'data/models/normalization_stats.npz'

model_data = np.load(MODEL_PATH)
WEIGHTS = model_data['weights']
BIASES = model_data['biases']

stats_data = np.load(STATS_PATH)
MEAN = stats_data['mean']
STD = stats_data['std']

processor = ImageProcessor()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """Reproduit exactement le pipeline utilisé pendant l'entraînement :
    extraction des features -> normalisation avec mean/std du training -> softmax"""
    features = processor.extract_features(image_path)

    # Normalisation avec les stats sauvegardées (PAS un nouveau fit !)
    features_normalized = (features - MEAN) / (STD + 1e-8)

    scores = np.dot(WEIGHTS, features_normalized) + BIASES
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])

    class_probs = {
        processor.class_names[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(processor.class_names))
    }

    return {
        'class_name': processor.class_names[predicted_class],
        'confidence': round(confidence * 100, 2),
        'all_probs': class_probs
    }


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="Aucune image envoyée.")

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error="Aucun fichier sélectionné.")

    if not allowed_file(file.filename):
        return render_template('index.html', error="Format non supporté (png, jpg, jpeg uniquement).")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_image(filepath)

    return render_template(
        'index.html',
        result=result,
        image_path=filepath
    )


if __name__ == '__main__':
    app.run(debug=True)
