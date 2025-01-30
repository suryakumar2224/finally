import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize webcam
cap = cv2.VideoCapture(0)

# Hand detector (detects only one hand at a time)
detector = HandDetector(maxHands=1)

# Image processing parameters
offset = 20
imgSize = 300
counter = 0

# Gesture categories
gestures = ["Okay", "ThumbsUp", "Peace", "Stop"]
gesture_index = 0  # Default gesture selection
folder = f"Data/{gestures[gesture_index]}"

# Ensure directories exist
for gesture in gestures:
    os.makedirs(f"Data/{gesture}", exist_ok=True)

print("Press 'n' to switch gestures | 's' to save | 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        continue

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create white canvas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region
        try:
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            imgCropShape = imgCrop.shape

            # Maintain aspect ratio
            aspectRatio = h / w

            if aspectRatio > 1:  # Tall Image
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:  # Wide Image
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Display cropped images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        except Exception as e:
            print("Error in cropping:", e)

    # Display the main camera feed
    cv2.putText(img, f"Gesture: {gestures[gesture_index]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Image', img)

    # Key handling
    key = cv2.waitKey(1)

    if key == ord("s"):  # Save image
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved {counter} images for {gestures[gesture_index]}")

    elif key == ord("n"):  # Switch gesture
        gesture_index = (gesture_index + 1) % len(gestures)
        folder = f"Data/{gestures[gesture_index]}"
        print(f"Switched to: {gestures[gesture_index]}")

    elif key == ord("q"):  # Quit
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
