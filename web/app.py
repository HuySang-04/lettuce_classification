from flask import Flask, render_template, request, jsonify, url_for
from tensorflow import keras
import numpy as np
from PIL import Image
import os, random

app = Flask(__name__)

# model = keras.models.load_model("lettuce_health_VGG19_classification.keras")
model = keras.models.load_model("/home/sang/Downloads/model_save/InceptionV3_lettuce_health_classification.keras")
model2 = keras.models.load_model("lettuce_disease_vgg19_classification_3.keras")

# CLASS_NAMES = ['unhealthy', 'healthy']
CLASS_NAMES = ['healthy', 'unhealthy']

CLASS_NAMES_2 = [
    'Downy_mildew',
    'Anthracnose',
    'Bacterial',
    'Powdery_mildew',
    'Viral',
    'Septoria_blight'
]

EN_TO_VI = {
    'Downy_mildew': 'Bệnh sương mai',
    'Anthracnose': 'Bệnh thán thư',
    'Bacterial': 'Bệnh do vi khuẩn',
    'Powdery_mildew': 'Bệnh phấn trắng',
    'Viral': 'Bệnh do virus',
    'Septoria_blight': 'Bệnh Septoria trên xà lách'
}

DISEASE_FOLDER_MAP = {
    'Bệnh sương mai': 'Downy_mildew',
    'Bệnh thán thư': 'Anthracnose',
    'Bệnh do vi khuẩn': 'Bacterial',
    'Bệnh phấn trắng': 'Powdery_mildew',
    'Bệnh do virus': 'Viral',
    'Bệnh Septoria trên xà lách': 'Septoria_blight'
}


def get_random_images(disease_name, n=2):
    if not disease_name:
        return []

    disease_name = disease_name.strip()

    if disease_name in DISEASE_FOLDER_MAP.values():
        folder = disease_name
    elif disease_name in DISEASE_FOLDER_MAP:
        folder = DISEASE_FOLDER_MAP[disease_name]
    else:
        candidate = disease_name.replace(" ", "_")
        if candidate in DISEASE_FOLDER_MAP.values():
            folder = candidate
        else:
            return []

    folder_path = os.path.join("static", "suggestion_images", folder)
    if not os.path.exists(folder_path):
        return []

    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        return []

    selected = random.sample(images, min(n, len(images)))
    return [url_for('static', filename=f'suggestion_images/{folder}/{img}')
            for img in selected]

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

    if main_class == 'healthy':
        return jsonify({'class': 'healthy'})

    preds_2 = model2.predict(img_array)[0]
    confidence = float(np.max(preds_2))
    top_idx = np.argsort(preds_2)[::-1]

    if confidence < 0.8:
        top2_idx = top_idx[:2]
        results = []
        for i in top2_idx:
            name = CLASS_NAMES_2[i]
            conf = round(float(preds_2[i]) * 100, 2)
            results.append({
                'name': name,
                'name_vi': EN_TO_VI.get(name, name),
                'confidence': conf,
                'suggestions': get_random_images(name, n=2)
            })

        return jsonify({
            'class': 'unhealthy',
            'detail': results
        })

    else:
        top_class = CLASS_NAMES_2[top_idx[0]]
        return jsonify({
            'class': 'unhealthy',
            'detail': [{
                'name': top_class,
                'name_vi': EN_TO_VI.get(top_class, top_class),
                'confidence': round(confidence * 100, 2),
                'suggestions': get_random_images(top_class, n=2)
            }]
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
