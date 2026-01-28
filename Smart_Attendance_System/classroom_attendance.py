import cv2
import face_recognition
import pickle
import pandas as pd
import time

# -------- CONFIGURATION --------
SUBJECTS = ["Sub1", "Sub2", "Sub3", "Sub4", "Sub5", "Sub6", "Sub7"]
PERIOD_TIME = 60  # minutes per subject
# --------------------------------

students_df = pd.read_csv("students.csv")
roll_to_name = dict(zip(students_df["RollNo"], students_df["Name"]))
roll_to_section = dict(zip(students_df["RollNo"], students_df["Section"]))

with open("trained_faces.pkl", "rb") as f:
    known_encodings, known_rollnos = pickle.load(f)

attendance_counter = {
    roll: {sub: 0 for sub in SUBJECTS}
    for roll in roll_to_name
}

cap = cv2.VideoCapture(0)

for subject in SUBJECTS:
    print(f"Scanning entire classroom for {subject}")
    start_time = time.time()

    while (time.time() - start_time) < PERIOD_TIME * 60:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for encoding, loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                known_encodings, encoding, tolerance=0.5
            )

            label = "Unknown"
            if True in matches:
                idx = matches.index(True)
                roll = known_rollnos[idx]
                label = roll
                attendance_counter[roll][subject] += 1

            top, right, bottom, left = [v * 2 for v in loc]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Smart Classroom Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# -------- FINAL EXCEL OUTPUT --------
rows = []

for roll in roll_to_name:
    row = [roll, roll_to_name[roll], roll_to_section[roll]]
    subject_percentages = []

    for sub in SUBJECTS:
        present_minutes = attendance_counter[roll][sub] / 60
        percent = round((present_minutes / PERIOD_TIME) * 100, 2)
        subject_percentages.append(percent)
        row.append(percent)

    overall_percent = round(sum(subject_percentages) / len(subject_percentages), 2)
    row.append(overall_percent)
    rows.append(row)

columns = ["Roll No", "Name", "Section"]
columns += [f"{sub} %" for sub in SUBJECTS]
columns.append("Overall %")

df = pd.DataFrame(rows, columns=columns)
df.to_excel("attendance.xlsx", index=False)

print("Final attendance saved in attendance.xlsx")
