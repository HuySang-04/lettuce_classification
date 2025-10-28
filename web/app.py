from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image

app = Flask(__name__)

model = keras.models.load_model("../outputs/InceptionV3_lettuce_health_classification.keras")

CLASS_NAMES = ['healthy', 'unhealthy']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = int(preds[0][0] > 0.5)
    main_class = CLASS_NAMES[pred_idx]

    return jsonify({
        'class': main_class,
        'confidence': round(float(preds[0][0] if main_class == 'unhealthy' else 1 - preds[0][0]), 4)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)