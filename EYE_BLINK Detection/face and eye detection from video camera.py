#First of all we are importing all the necessary library


import cv2 #OpenCV) is used for image and video processing.
import dlib #provides face detection and facial landmark detection.
import imutils #helps with image resizing and other utility functions.
from scipy.spatial import distance as dist # is used to compute Euclidean distances.
from imutils import face_utils #provides facial landmark indices.

# Set variables for blink detection
blink_threshold = 0.5 #is the threshold for the Eye Aspect Ratio (EAR) below which a blink is detected.
frame_success = 2 #is the number of consecutive frames where the EAR is below the threshold to confirm a blink.
frame_count = 0 #keeps track of the number of frames with EAR below the threshold.



# Initialize a video capture object to capture video from the default camera

cam = cv2.VideoCapture(0) #Opens the default camera (camera index 0) for capturing video.




# Access the indices for the left and right eyes using the dictionary keys
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #L_start and L_end are the indices for the left eye landmarks.

(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] #R_start and R_end are the indices for the right eye landmarks.

# Initialize the face detector from dlib
detector = dlib.get_frontal_face_detector() #detector is a face detector from dlib.

# Read the pre-trained model from the local file
predictor_path = 'D:/projects/EYE detection/shape_predictor_68_face_landmarks.dat' #predict is a facial landmark predictor loaded from a pre-trained .dat file.
predict = dlib.shape_predictor(predictor_path)


# This Python function calculates the Eye Aspect Ratio (EAR) of a given eye landmark set. 
# It computes the distance between the vertical eye landmarks and the distance between
#  the horizontal eye landmarks using the Euclidean distance formula. 
#  Then, it calculates the EAR by dividing the sum of vertical distances by the
#  horizontal distance. The function returns the EAR value as output.
def EAR_calculate(eye):
    a1 = dist.euclidean(eye[1], eye[5]) #a1 and a2 are the vertical distances between eye landmarks.
    a2 = dist.euclidean(eye[2], eye[4]) 
    m = dist.euclidean(eye[0], eye[3]) #m is the horizontal distance between eye landmarks.
    EAR = (a1 + a2) / m #EAR is calculated by dividing the sum of vertical distances by the horizontal distance.
    return EAR



# This function draws lines and circles on an image to mark the eye landmarks,
#  which can be useful for visualizing the detected landmarks from  video camera.

def eyeLandmark(frame, eyes):
    for eye in eyes:
        x1, x2 = (eye[1], eye[5])
        x3, x4 = (eye[0], eye[3])
        cv2.line(frame, x1, x2, (178, 200, 226), 2)
        cv2.line(frame, x3, x4, (178, 200, 226), 2)
        for i in range(6):
            cv2.circle(frame, tuple(eye[i]), 3, (200, 223, 0), -1)
    return frame

#start the loop
while True:       #Captures each frame from the camera.
    ret, frame = cam.read()
    if not ret:
        break


#Resizes the frame and converts it to grayscale for face detection.
    frame = imutils.resize(frame, width=512) #esizes the frame to a width of 512 pixels while maintaining the aspect ratio.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converts the frame from BGR (color) to grayscale
    faces = detector(gray) #detect faces in the grayscale image

    for face in faces:   #Iterates over each detected face.
        shape = predict(gray, face) # to detect facial landmarks for the current face.
        shape = face_utils.shape_to_np(shape) # Converts the detected landmarks to a NumPy array for easier manipulation.
        
        for lm in shape: #Iterates over each facial landmark
            cv2.circle(frame, tuple(lm), 3, (10, 2, 200)) #Draws a small circle on the frame at each facial landmark position to visualize the landmarks.
        
        lefteye = shape[L_start:L_end] #Extracts the landmarks for the left eye based on pre-defined indices.
        righteye = shape[R_start:R_end]  #Extracts the landmarks for the right eye based on pre-defined indices.          
        
        left_EAR = EAR_calculate(lefteye) #Calculates the Eye Aspect Ratio (EAR) for the left eye using the EAR_calculate function.
        right_EAR = EAR_calculate(righteye) #Calculates the Eye Aspect Ratio (EAR) for the right eye using the EAR_calculate function.
        
        img = frame.copy() #Creates a copy of the original frame for further processing (to preserve the original).
        img = eyeLandmark(img, [lefteye, righteye]) #Draws eye landmarks on the frame using the eyeLandmark function.
        avg = (left_EAR + right_EAR) / 2 #Computes the average EAR of both eyes.
        
        if avg < blink_threshold: #Checks if the average EAR is below the blink threshold.
            frame_count += 1   # Increments the frame_count if the EAR is below the threshold, indicating a potential blink.
        else:
            if frame_count >= frame_success:
                # Checks if the number of consecutive frames with a low EAR meets the requirement for detecting a blink.
                #Display blink Detected
                cv2.putText(img, 'Blink Detected', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (233, 0, 189), 1)
            frame_count = 0
        #Shows the processed frame in a window named "Your Eye Blink".
    cv2.imshow("You_Eye_Blink", img)
    
    
 #Exits the loop and releases the camera if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
