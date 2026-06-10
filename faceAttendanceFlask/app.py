from flask import Flask, render_template, Response
import cv2
import csv
import os
from datetime import datetime

app = Flask(__name__)

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

attendance_marked = False

camera = cv2.VideoCapture(0)

def generate_frames():

    global attendance_marked

    while True:

        success, frame = camera.read()

        if not success:
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

        # No face detected
        if len(faces) == 0:

            cv2.putText(
                frame,
                "No Face Found",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
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

                    current_time = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    with open(
                        attendance_file,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            name,
                            current_time,
                            1
                        ])

                    print(
                        f"{name} attendance marked at {current_time}"
                    )

                    attendance_marked = True

                display_text = (
                    f"{name} - Attendance Marked"
                )

            else:

                display_text = "Unknown Person"

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

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

@app.route("/")
def index():

    return render_template(
        "index.html"
    )

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":

    app.run(
        debug=True
    )