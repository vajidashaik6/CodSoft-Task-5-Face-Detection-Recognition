from flask import Flask, render_template, request, send_from_directory
import cv2
import os
import json

MODEL_PATH = "trainer.yml"
LABELS_PATH = "labels.json"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open(LABELS_PATH, "r") as file:
    label_names = json.load(file)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


CASCADE_PATH = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(
        "Haar Cascade file not found: " + CASCADE_PATH
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image."

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(
        OUTPUT_FOLDER,
        "detected_" + file.filename
    )

    file.save(input_path)

    image = cv2.imread(input_path)

    if image is None:
        return "Invalid image."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    recognized_names = []

    for (x, y, w, h) in faces:

        face = gray[y:y + h, x:x + w]

        label_id, confidence = recognizer.predict(face)

        # LBPH confidence: lower value means better match
        if confidence < 80:
            name = label_names.get(str(label_id), "Unknown")
        else:
            name = "Unknown"

        recognized_names.append(name)

        # Draw face rectangle
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display recognized name
        cv2.putText(
            image,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_path, image)

    return render_template(
        "index.html",
        result_image="/outputs/detected_" + file.filename,
        face_count=len(faces),
        recognized_names=recognized_names
    )
    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image."

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, "detected_" + file.filename)

    file.save(input_path)

    image = cv2.imread(input_path)

    if image is None:
        return "Invalid image."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_path, image)

    return render_template(
        "index.html",
        result_image="/outputs/detected_" + file.filename,
        face_count=len(faces)
    )


@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)