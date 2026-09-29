import cv2
import os
import json

DATASET_PATH = "dataset"
MODEL_PATH = "trainer.yml"
LABELS_PATH = "labels.json"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError("Haar Cascade XML could not be loaded.")

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}

current_id = 0

# Read each person folder
for person_name in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    label_names[current_id] = person_name

    print(f"Training: {person_name}")

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in detected_faces:
            face = gray[y:y + h, x:x + w]

            faces.append(face)
            labels.append(current_id)

    current_id += 1

if len(faces) == 0:
    raise RuntimeError(
        "No faces found in dataset. Add clear face images and try again."
    )

# Train the model
recognizer.train(faces, __import__("numpy").array(labels))

# Save trained model
recognizer.write(MODEL_PATH)

# Save label names
with open(LABELS_PATH, "w") as file:
    json.dump(label_names, file)

print()
print("Training completed successfully!")
print(f"Faces used for training: {len(faces)}")
print(f"Model saved as: {MODEL_PATH}")
print(f"Labels saved as: {LABELS_PATH}")