import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Create folder if it doesn't exist
save_folder = "captured_faces"
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)

img_count = 0

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow(
        "Face Detection",
        frame
    )

    key = cv2.waitKey(1)

    # Press S to save image
    if key & 0xFF == ord('s'):

        filename = os.path.join(
            save_folder,
            f"face_{img_count}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"Image Saved: {filename}"
        )

        img_count += 1

    # Press Q to quit
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()