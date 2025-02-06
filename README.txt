Stroke Detection and Classification Project



Table of Contents

1. Project Overview

2. Folder Structure

3. Requirements

4. Dataset Preparation

5. Model Training

6. Running the Flask Application

7. Usage

8. Progress


---

1. Project Overview

This project is a web-based application designed to detect and classify stroke images using two pre-trained deep learning models:

Model 1: Classifies an image as either showing signs of a stroke or not.

Model 2: If a stroke is detected, this model further classifies the stroke as either hemorrhagic or ischaemic.


The web interface, built with Flask, allows users to upload an image. The app then uses the models to identify whether the image is normal or shows a stroke and, if a stroke is present, identifies its type.


---

2. Folder Structure

The project folder is organized as follows:

project_folder/
│
├── app.py                           # Flask application backend
├── stroke_detection_model.h5         # Trained model to classify stroke vs. normal
├── stroke_differentiate_model.keras  # Trained model to classify stroke type
├── dataset/                          # Folder for dataset images
│   ├── hemorrhagic/                  # Folder containing hemorrhagic stroke images
│   └── ischaemic/                    # Folder containing ischaemic stroke images
│   └── normal/                       # Folder containing normal (non-stroke) images
├── templates/                        # HTML templates for Flask
│   ├── index.html                    # Main upload page
│   └── result.html                   # Result display page
└── static/                           # Static files (CSS, JavaScript)
    └── styles.css                    # CSS for front-end styling


---

3. Requirements

To run this project, you need to install the following Python packages:

pip install tensorflow flask opencv-python-headless numpy matplotlib

Software Requirements

Python 3.7 

Flask (for the web interface)

TensorFlow (for loading and using the models)

OpenCV (for image preprocessing)

NumPy (for numerical operations)

Matplotlib (for visualizing model training if needed)



---

4. Dataset Preparation

Prepare your dataset by organizing images into separate folders based on their category:

dataset/hemorrhagic/: Contains images of hemorrhagic strokes.

dataset/ischaemic/: Contains images of ischaemic strokes.

dataset/normal/: Contains images of normal cases (required if the first model distinguishes stroke vs. non-stroke).


Ensure each subfolder contains images that are correctly labeled for their respective categories. This structure allows the models to load and train on the data efficiently.


---

5. Model Training

A. Train the First Model (Stroke Detection):

This model classifies an image as either "normal" or "stroke."

Save this model as stroke_detection_model.h5.



B. Train the Second Model (Stroke Differentiation):

This model further classifies stroke images as either "hemorrhagic" or "ischaemic."

Save this model as stroke_differentiate_model.keras.




The dataset/ folder should be structured as described above for proper loading and processing by the training script. Ensure each model is saved in the root directory of the project folder (project_folder/) as stroke_detection_model.h5 and stroke_differentiate_model.keras, respectively.


---

6. Running the Flask Application

A. Start the Flask Server: From the project directory, run the following command to start the Flask application:

python app.py


B. Access the Web Interface: Open a browser and go to http://127.0.0.1:5000 to access the main upload page.




---

7. Usage

A. Upload an Image:

On the main page, use the form to upload an image file.

The image should ideally be a medical scan suitable for stroke classification (e.g., CT or MRI).



B. View Results:

After uploading, the app processes the image through the two models.

The first model detects if the image shows signs of a stroke.

If a stroke is detected, the second model classifies the stroke type as hemorrhagic or ischaemic.

Results are displayed on a separate page (result.html), with an option to return to the main page for further uploads.





---
