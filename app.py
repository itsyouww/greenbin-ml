from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import io  

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load model sekali saat startup
model = load_model('deteksi_klasifikasi/model/model_sampah.h5')
class_names = ['cardboard', 'metal', 'paper', 'plastic', 'trash']  # sesuaikan dengan labelmu

def preprocess_image(file):
    try:
        image = load_img(io.BytesIO(file.read()), target_size=(150, 150))  # ✅ perbaikan di sini
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image
    except Exception as e:
        print(f"Error saat memproses gambar: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = preprocess_image(file)
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        prediction = model.predict(image)
        label = class_names[np.argmax(prediction)]
        return jsonify({'hasil': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # `host='0.0.0.0'` agar bisa diakses dari luar
