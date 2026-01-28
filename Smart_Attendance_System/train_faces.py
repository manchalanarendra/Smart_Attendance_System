import face_recognition
import os
import pickle

DATASET_DIR = "dataset"

known_encodings = []
known_rollnos = []

for roll_no in os.listdir(DATASET_DIR):
    student_path = os.path.join(DATASET_DIR, roll_no)
    if not os.path.isdir(student_path):
        continue

    for img in os.listdir(student_path):
        img_path = os.path.join(student_path, img)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_rollnos.append(roll_no)

with open("trained_faces.pkl", "wb") as f:
    pickle.dump((known_encodings, known_rollnos), f)

print("Face training completed for entire class")
