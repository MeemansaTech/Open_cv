import cv2

# Load the pre-trained face and eye detector models (Haar Cascades)
face_classifier = cv2.CascadeClassifier(r"D:\projects\HAND & EYE DETECTION\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(r"D:\projects\HAND & EYE DETECTION\haarcascade_eye.xml")

# Open the default camera (usually the first camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally to act as a mirror
    frame = cv2.flip(frame, 1)

    # Resize the frame to fit within the screen dimensions (optional)
    screen_width = 800  # You can set this to your screen width
    screen_height = 600  # You can set this to your screen height
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    if frame_width > screen_width or frame_height > screen_height:
        if aspect_ratio > 1:
            # Width is greater, resize based on width
            new_width = screen_width
            new_height = int(screen_width / aspect_ratio)
        else:
            # Height is greater, resize based on height
            new_height = screen_height
            new_width = int(screen_height * aspect_ratio)
        frame = cv2.resize(frame, (new_width, new_height))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around each face and detect eyes within the face region
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    # Display the output frame
    cv2.imshow('Face and Eye Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
