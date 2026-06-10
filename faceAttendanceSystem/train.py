import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
names = {}

label_id = 0

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(
        dataset_path,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue

    names[label_id] = person_name

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        detected_faces = face_detector.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in detected_faces:

            face = image[
                y:y+h,
                x:x+w
            ]

            face = cv2.resize(
                face,
                (200, 200)
            )

            faces.append(face)
            labels.append(label_id)

    label_id += 1

print("Faces collected:", len(faces))

if len(faces) == 0:
    print("No faces found in dataset!")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(labels)
)

recognizer.save(
    "trainer.yml"
)

with open(
    "labels.txt",
    "w"
) as f:

    for key, value in names.items():
        f.write(
            f"{key},{value}\n"
        )

print("Training completed!")