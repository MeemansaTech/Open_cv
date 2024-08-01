import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import csv
import time  # For logging timestamps

# Set variables for blink detection
blink_threshold = 0.5
frame_success = 2
frame_count = 0

# Initialize a video capture object to capture video from the default camera
cam = cv2.VideoCapture(0)

# Access the indices for the left and right eyes using the dictionary keys
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize the face detector from dlib
detector = dlib.get_frontal_face_detector()

# Read the pre-trained model from the local file
predictor_path = 'D:/projects/EYE detection/shape_predictor_68_face_landmarks.dat'
predict = dlib.shape_predictor(predictor_path)

# Function to calculate the Eye Aspect Ratio (EAR)
def EAR_calculate(eye):
    a1 = dist.euclidean(eye[1], eye[5])
    a2 = dist.euclidean(eye[2], eye[4])
    m = dist.euclidean(eye[0], eye[3])
    EAR = (a1 + a2) / m
    return EAR

# Function to draw eye landmarks on the frame
def eyeLandmark(frame, eyes):
    for eye in eyes:
        x1, x2 = (eye[1], eye[5])
        x3, x4 = (eye[0], eye[3])
        cv2.line(frame, x1, x2, (178, 200, 226), 2)
        cv2.line(frame, x3, x4, (178, 200, 226), 2)
        for i in range(6):
            cv2.circle(frame, tuple(eye[i]), 3, (200, 223, 0), -1)
    return frame

# Open a CSV file to log blink detection
with open('blink_detection_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Blink Detected'])  # Write the header row

    # Start the loop
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=512)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predict(gray, face)
            shape = face_utils.shape_to_np(shape)

            for lm in shape:
                cv2.circle(frame, tuple(lm), 3, (10, 2, 200))

            lefteye = shape[L_start:L_end]
            righteye = shape[R_start:R_end]

            left_EAR = EAR_calculate(lefteye)
            right_EAR = EAR_calculate(righteye)

            img = frame.copy()
            img = eyeLandmark(img, [lefteye, righteye])
            avg = (left_EAR + right_EAR) / 2

            if avg < blink_threshold:
                frame_count += 1
            else:
                if frame_count >= frame_success:
                    cv2.putText(img, 'Blink Detected', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (233, 0, 189), 1)
                    # Log the blink detection to the CSV file with a timestamp
                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), 'Yes'])
                frame_count = 0

        cv2.imshow("You_Eye_Blink", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
