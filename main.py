from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# --- Load the two models from the root directory where they will be downloaded ---
try:
    stroke_detection_model = tf.keras.models.load_model('stroke_detection_model.h5')
    stroke_type_model = tf.keras.models.load_model('stroke_differentiate_model.keras')
except Exception as e:
    # This will help debug if models fail to load
    print(f"Error loading models: {e}")
    stroke_detection_model = None
    stroke_type_model = None


# Define function to preprocess and classify the image
def predict_stroke_type(image):
    image = cv2.resize(image, (128, 128)) / 255.0
    image = image.reshape(-1, 128, 128, 1)

    stroke_pred = stroke_detection_model.predict(image)
    stroke_label = np.argmax(stroke_pred, axis=1)[0]

    # Assuming label 1 is 'Normal' based on your original code
    if stroke_label == 1:
        return "Normal", None

    stroke_type_pred = stroke_type_model.predict(image)
    stroke_type_label = np.argmax(stroke_type_pred, axis=1)[0]

    stroke_types = {0: "Hemorrhagic", 1: "Ischemic"}
    return "Stroke", stroke_types.get(stroke_type_label, "Unknown Type")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not all([stroke_detection_model, stroke_type_model]):
         return jsonify({"error": "Models are not loaded on the server."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Process image in memory
            in_memory_file = np.fromstring(file.read(), np.uint8)
            image = cv2.imdecode(in_memory_file, cv2.IMREAD_GRAYSCALE)

            if image is None:
                return jsonify({"error": "Invalid image format"}), 400

            # Get the predictions
            stroke_status, stroke_type = predict_stroke_type(image)

            return jsonify({"status": stroke_status, "type": stroke_type})
        except Exception as e:
            return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

    return jsonify({"error": "An unknown error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
