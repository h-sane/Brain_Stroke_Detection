from flask import Flask, render_template, request, jsonify
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import os

app = Flask(__name__)

# --- Load the TFLite model ---
base_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(base_dir, 'model.tflite')
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Read and preprocess the image in memory
            in_memory_file = np.fromstring(file.read(), np.uint8)
            image = cv2.imdecode(in_memory_file, cv2.IMREAD_GRAYSCALE)

            # Resize to the model's expected input size (128x128)
            resized_image = cv2.resize(image, (128, 128))
            
            # Prepare image for the model: add batch and channel dimensions, and normalize
            input_data = np.expand_dims(resized_image, axis=-1) # Add channel dimension
            input_data = np.expand_dims(input_data, axis=0) # Add batch dimension
            input_data = input_data.astype(np.float32) / 255.0 # Normalize

            # Set the input tensor, invoke the interpreter, and get the result
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Process the output
            prediction_label = np.argmax(output_data[0])
            status = "Stroke" if prediction_label == 0 else "Normal"
            
            return jsonify({"status": status})

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    return jsonify({"error": "An unknown error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
