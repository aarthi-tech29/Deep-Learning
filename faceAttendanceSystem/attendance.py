import cv2
import csv
import os
from datetime import datetime

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load labels
names = {}

with open("labels.txt", "r") as f:
    for line in f:
        label, name = line.strip().split(",")
        names[int(label)] = name

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

attendance_file = "attendance.csv"

# Create attendance file if not exists
if not os.path.exists(attendance_file):

    with open(
        attendance_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Name",
            "DateTime",
            "Attendance"
        ])

# Mark only once per webcam session
attendance_marked = False

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face_roi = gray[
            y:y+h,
            x:x+w
        ]

        face_roi = cv2.resize(
            face_roi,
            (200, 200)
        )

        label, confidence = recognizer.predict(
            face_roi
        )

        if confidence < 60:

            name = names[label]

            if not attendance_marked:

                with open(
                    attendance_file,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        name,
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        1
                    ])

                print(
                    f"{name} attendance marked"
                )

                attendance_marked = True

            display_text = (
                f"{name} - Attendance Marked"
            )

        else:

            display_text = "Unknown"

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            display_text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Attendance System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()