from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load models
stroke_detection_model = tf.keras.models.load_model('stroke_detection_model.h5')
stroke_type_model = tf.keras.models.load_model('stroke_differentiate_model.keras')


# Define function to preprocess and classify the image
def predict_stroke_type(image):
    image = cv2.resize(image, (128, 128)) / 255.0
    image = image.reshape(-1, 128, 128, 1)

    stroke_pred = stroke_detection_model.predict(image)
    stroke_label = np.argmax(stroke_pred, axis=1)[0]

    if stroke_label == 1:  # 'Normal'
        return "Normal", None

    stroke_type_pred = stroke_type_model.predict(image)
    stroke_type_label = np.argmax(stroke_type_pred, axis=1)[0]

    stroke_types = {0: "Hemorrhagic", 1: "Ischemic"}
    return "Stroke", stroke_types[stroke_type_label]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Load and preprocess the image
    file_path = os.path.join("static", file.filename)
    file.save(file_path)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Get the predictions
    stroke_status, stroke_type = predict_stroke_type(image)

    # Delete the image after processing (cleanup)
    os.remove(file_path)

    return jsonify({"status": stroke_status, "type": stroke_type})


if __name__ == '__main__':
    app.run(debug=True)
