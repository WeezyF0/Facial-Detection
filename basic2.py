import cv2

def detect_and_display(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        face_roi = frame_gray[y:y + h, x:x + w]

        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(face_roi)
        for (ex, ey, ew, eh) in eyes:
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            radius = round((ew + eh) * 0.25)
            cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

    # Show the result
    cv2.imshow("Capture - Face detection", frame)

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    camera_device = 0
    capture = cv2.VideoCapture(camera_device)
    if not capture.isOpened():
        print("--(!) Error opening video capture")
        exit()

    while True:
        ret, frame = capture.read()
        if not ret or frame is None:
            print("--(!) No captured frame -- Break!")
            break

        detect_and_display(frame)

        if cv2.waitKey(10) == 27:
            break  # Press 'Esc' to exit

    capture.release()
    cv2.destroyAllWindows()
