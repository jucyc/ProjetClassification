import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.image_processor import ImageProcessor
from python_api.rbf_bridge import RBFModel

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== CHARGEMENT DU MODELE ====================

MODEL_PATH = 'data/models/rbf_model.bin'
MEAN_PATH  = 'data/models/rbf_mean.npy'
STD_PATH   = 'data/models/rbf_std.npy'

processor = ImageProcessor()
model = RBFModel(n_centers=1, n_features=1024, n_classes=3, gamma=0.001)
model.load(MODEL_PATH)

MEAN = np.load(MEAN_PATH)
STD  = np.load(STD_PATH)

print("Modele RBF et stats de normalisation charges.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def softmax(scores):
    """Convertit les scores bruts en pourcentages (somme = 100%)."""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def predict_image(image_path):
    features = processor.extract_features(image_path)
    features_normalized = (features - MEAN) / (STD + 1e-8)
    pred_class = model.predict(list(features_normalized))
    return {
        'class_name': processor.class_names[pred_class]
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