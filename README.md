# Face Detection and Recognition AI

## CodeSoft Task 5

An AI-powered computer vision application developed using Python, OpenCV, Haar Cascade and LBPH Face Recognition.

The application can detect faces in uploaded images and recognize locally enrolled test faces.

---

## 🎯 Objective

The main objective of this project is to develop an AI application that can:

- Detect human faces in images.
- Recognize enrolled faces.
- Mark unknown faces as "Unknown".
- Display detected faces with bounding boxes.
- Provide a simple and professional web interface.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Haar Cascade Classifier
- LBPH Face Recognizer
- Flask
- HTML
- CSS
- NumPy

---

## 📂 Project Structure

```text
Face_Detection_Recognition
│
├── app.py
├── face_detection.py
├── train_model.py
├── trainer.yml
├── labels.json
├── requirements.txt
├── haarcascade_frontalface_default.xml
│
├── dataset
│   └── Vajida
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
│
├── uploads
├── outputs
│
├── static
│   └── style.css
│
├── templates
│   └── index.html
│
└── venv


🚀 Installation
Step 1: Create Virtual Environment
python -m venv venv
Step 2: Activate Virtual Environment

Windows:

venv\Scripts\activate
Step 3: Install Required Libraries
python -m pip install -r requirements.txt
▶️ Run the Application

First train the model:

python train_model.py

Then start the Flask application:

python app.py

Open the following address in the browser:

http://127.0.0.1:5000