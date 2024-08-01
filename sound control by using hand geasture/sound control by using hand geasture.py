import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing Utility
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        # Convert the image color back to BGR
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks and highlight fingertips
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the landmarks for the thumb and index finger
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Highlight fingertips
                thumb_center = (int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0]))
                index_center = (int(index.x * frame.shape[1]), int(index.y * frame.shape[0]))
                cv2.circle(frame, thumb_center, 10, (0, 255, 0), -1)
                cv2.circle(frame, index_center, 10, (0, 255, 0), -1)

                # Draw a line between the centers of the thumb and index finger circles
                cv2.line(frame, thumb_center, index_center, (0, 255, 0), 2)

                # Calculate the distance between the thumb and index finger,
                distance = math.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2)

                # Control the volume based on the distance
                if distance < 0.05:  # Adjust the threshold based on your hand size and camera resolution
                    pyautogui.press('volumedown')
                elif distance > 0.1:  # Adjust the threshold based on your hand size and camera resolution
                    pyautogui.press('volumeup')

        # Display the frame
        cv2.imshow('Silent Knights Hands', frame)

        if cv2.waitKey(5) & 0xFF ==ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
