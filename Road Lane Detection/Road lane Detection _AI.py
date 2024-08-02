import tkinter as tk #We use it to create a graphical user interface.
from tkinter import *
import cv2 #Part of the OpenCV library for computer vision tasks.

from PIL import Image, ImageTk # Image and ImageTk modules for handling image conversion and displaying.
import numpy as np # for numeric computation



#Declare Global Variable

global last_frame1  # Declaring last_frame1 as a global variable to store the last frame from video capture.
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)  # Initialize last_frame1 as a blank black frame with dimensions 480x640 and 3 color channels (RGB).

global last_frame2  # Declaring last_frame2 as a global variable to store the last processed frame.
last_frame2 = np.zeros((480, 640, 3), dtype=np.uint8)  # Initialize last_frame2 similarly to last_frame1.




#Captures the video from the specified file path.
global cap1
cap1 = cv2.VideoCapture(r"D:\projects\19.Road Lane Detection\test_video.mp4")




def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the input frame to grayscale.
    
    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur to the grayscale image to smooth it and reduce noise.
    
    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)  # Perform Canny edge detection on the blurred image to find edges.
    
    # Define a region of interest
    height, width = frame.shape[:2]  # Get the dimensions of the frame (height and width).
    mask = np.zeros_like(edges)  # Create a mask with the same dimensions as the edges image, initialized to zero.
    
    # Define a polygonal region of interest (ROI)
    polygon = np.array([[
        (0, height),  # Bottom-left corner of the ROI
        (width, height),  # Bottom-right corner of the ROI
        (width, height // 2),  # Top-right corner of the ROI
        (0, height // 2)  # Top-left corner of the ROI
    ]], np.int32)  # Create a polygon array defining the ROI.
    
    cv2.fillPoly(mask, polygon, 255)  # Fill the polygon on the mask with white color (255).
    cropped_edges = cv2.bitwise_and(edges, mask)  # Apply the mask to the edges image to keep only the region of interest.
    
    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, maxLineGap=50)  # Detect lines in the ROI using Hough Line Transform.
    
    # Draw lines on the original frame
    line_frame = np.copy(frame)  # Create a copy of the original frame for drawing lines.
    if lines is not None:  # Check if any lines were detected.
        for line in lines:  # Iterate through each detected line.
            x1, y1, x2, y2 = line[0]  # Extract the coordinates of the line.
            cv2.line(line_frame, (x1, y1), (x2, y2), (17, 24, 118), 5)  # Draw the line on the frame in green with a thickness of 5 pixels.
    
    return line_frame  # Return the frame with the detected lines drawn on it.



#Video Display Function

def show_vid():
    if not cap1.isOpened(): # Check if the video capture object is opened.
        print("Can't open the camera1") # # Print an error message if the video capture could not be opened.
        return
    flag1, frame1 = cap1.read()  # # Read a frame from the video capture.
    if not flag1:  # Check if the frame was successfully read.
        print("Reached end of video or can't read the frame!")  ## Print an error message if the frame could not be read.
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video from the begining
        flag1, frame1 = cap1.read() #try again a frame after restarting
    if not flag1:  ## Check again if the frame was successfully read.
        return
    frame1 = cv2.resize(frame1, (400, 500))  ## Resize the frame to 400x500 pixels.
    global last_frame1
    last_frame1 = frame1.copy()
    pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_vid)
    
    
    
def show_vid2():
    if not cap1.isOpened():  # Check if the video capture object is opened.
        print("Can't open the camera2")  # Print an error message if the video capture could not be opened.
        return
    
    flag2, frame2 = cap1.read()  # Read a frame from the video capture.
    if not flag2:  # Check if the frame was successfully read.
        print("Reached end of video or can't read the frame!")  # Print an error message if the frame could not be read.
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video from the beginning.
        flag2, frame2 = cap1.read()  # Try reading a frame again after restarting.
    
    if not flag2:  # Check again if the frame was successfully read.
        return
    
    frame2 = cv2.resize(frame2, (400, 500))  # Resize the frame to 400x500 pixels.
    frame2 = process_frame(frame2)  # Process the frame to detect and draw lane lines.
    
    global last_frame2
    last_frame2 = frame2.copy()  # Update last_frame2 with the processed frame.
    
    pic2 = cv2.cvtColor(last_frame2, cv2.COLOR_BGR2RGB)  # Convert the processed frame to RGB color space.
    img2 = Image.fromarray(pic2)  # Convert the RGB image to a PIL Image object.
    img2tk = ImageTk.PhotoImage(image=img2)  # Convert the PIL Image to an ImageTk PhotoImage object for display in tkinter.
    
    lmain2.img2tk = img2tk  # Store the PhotoImage reference to prevent garbage collection.
    lmain2.configure(image=img2tk)  # Update the tkinter Label widget with the new image.
    lmain2.after(10, show_vid2)  # Schedule the next frame update after 10 milliseconds.



if __name__ == '__main__':  # Check if this script is being run as the main program.
    root = tk.Tk()  # Create the main tkinter window.
    lmain = tk.Label(master=root)  # Create a Label widget for displaying the original video frames.
    lmain2 = tk.Label(master=root)  # Create another Label widget for displaying the processed video frames.

    lmain.pack(side=LEFT)  # Pack the first Label widget on the left side of the window.
    lmain2.pack(side=RIGHT)  # Pack the second Label widget on the right side of the window.
    
    root.title("Lane-line detection")  # Set the title of the window.
    root.geometry("900x700+100+10")  # Set the size and position of the window.
    
    exitbutton = Button(root, text='Quit', fg="Blue", command=root.destroy).pack(side=BOTTOM,)  # Create a Quit button to close the application.
    
    show_vid()  # Start displaying the original video frames in the first Label widget.
    show_vid2()  # Start displaying the processed video frames in the second Label widget.
    
    root.mainloop()  # Start the tkinter event loop to run the application.
    cap1.release()  # Release the video capture object when the application is closed.
