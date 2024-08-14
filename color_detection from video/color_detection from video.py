import cv2
import pandas as pd

# Load the CSV file with color details
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv(r'D:\projects\21.Color Ddetection\colors.csv', names=index, header=None)

# Function to calculate the closest color name
def getColorName(R, G, B):
    minimum = float('inf')
    cname = ""
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d < minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

# Initialize global variables
clicked = False
r = g = b = xpos = ypos = 0

# Function to capture the coordinates and color on mouse movement
def draw_function(event, x, y, flags, param):
    global b, g, r, xpos, ypos, clicked
    if event == cv2.EVENT_MOUSEMOVE:
        clicked = True
        xpos = x
        ypos = y
        b, g, r = frame[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

# Access the camera feed
cap = cv2.VideoCapture(0)
cv2.namedWindow('Camera Feed')
cv2.setMouseCallback('Camera Feed', draw_function)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if clicked:
        # Draw a rectangle and display the color name and RGB values
        cv2.rectangle(frame, (20, 20), (750, 60), (b, g, r), -1)
        text = getColorName(r, g, b) + f' R={r} G={g} B={b}'
        cv2.putText(frame, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Display text in black if the color is too light
        if r + g + b >= 600:
            cv2.putText(frame, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
