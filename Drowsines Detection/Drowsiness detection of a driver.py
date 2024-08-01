# Importing libraries
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize pygame mixer
mixer.init()

# Load audio file
sound = mixer.Sound(r'D:\projects\Driver Drowsiness Detection\Drowsiness detection\alarm.wav')

# Load pretrained face and eye recognition model
face = cv2.CascadeClassifier(r'D:\projects\Driver Drowsiness Detection\Drowsiness detection\haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'D:\projects\Driver Drowsiness Detection\Drowsiness detection\haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'D:\projects\Driver Drowsiness Detection\Drowsiness detection\haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Closed', 'Open']
model = load_model(r'D:\projects\Driver Drowsiness Detection\Drowsiness detection\models\cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

# Variable to track the time when eyes were first detected as closed
closed_start_time = None
alarm_triggered = False

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
    
    # Reset the closed_start_time if at least one eye is open
    eyes_open = False
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            count += 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = model.predict(r_eye)
            rpred = np.argmax(rpred, axis=1)
            if rpred[0] == 1:
                lbl = 'Open'
                eyes_open = True
            elif rpred[0] == 0:
                lbl = 'Closed'
            break
    
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        lpred = np.argmax(lpred, axis=1)
        if lpred[0] == 1:
            lbl = 'Open'
            eyes_open = True
        elif lpred[0] == 0:
            lbl = 'Closed'
        break
    
    if not eyes_open:
        # Track the start time if both eyes are closed
        if closed_start_time is None:
            closed_start_time = time.time()
        else:
            # Check if eyes have been closed for more than 15 seconds
            if time.time() - closed_start_time > 15:
                if not alarm_triggered:
                    try:
                        sound.play()
                        alarm_triggered = True
                    except:
                        pass
    else:
        # Reset tracking if at least one eye is open
        closed_start_time = None
        alarm_triggered = False
    
    if closed_start_time and time.time() - closed_start_time <= 15:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    if score < 0:
        score = 0
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
    
    if thicc < 16:
        thicc += 2
    else:
        thicc -= 2
        if thicc < 2:
            thicc = 2
    
    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
