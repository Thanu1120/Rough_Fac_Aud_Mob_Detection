import cv2
import matplotlib.pyplot as plt
import numpy as np

# Your code for face and phone detection goes here...

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Object detection and other processing...

    # Convert BGR to RGB (matplotlib expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()

    # Break the loop (since matplotlib will block the loop, you'll need a manual exit)
    break

cap.release()
