import csv
import numpy as np
import cv2 as cv
import mediapipe as mp
from model import landmarkIdentifier
from PIL import Image, ImageDraw, ImageFont
import board
import adafruit_ssd1306

def initialise():
    global oled, draw, width, height, font, image
    
    i2c = board.I2C()

    oled = adafruit_ssd1306.SSD1306_I2C(128, 32, i2c)
    oled.fill(0)
    oled.show()

    width = oled.width
    height = oled.height
    image = Image.new("1", (width, height))
    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

def display_text(text):
    global oled, draw, width, height, font, image 
    
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    
    draw.text((0, 0), text, font=font, fill=255)

    oled.image(image)
    oled.show()

def main():
    cap = cv.VideoCapture(1)  # REMEMBER CHANGE TO 1 FOR RASPBERRY PI
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.80, min_tracking_confidence=0.55)

    landmark_identifier = landmarkIdentifier()

    with open('model/data/landmarkLabels.csv') as f:
        identifierLabels = csv.reader(f)
        identifierLabels = [
            row[0] for row in identifierLabels
        ]

    while True:
        success, frame = cap.read()
        if not success:
            break

        frameRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frameRgb.flags.writeable = False  # to optimise performance REMEMBER
        processing_results = hands.process(frameRgb)
        frameRgb.flags.writeable = True 

        if processing_results.multi_hand_landmarks:
            for hand_landmarks in processing_results.multi_hand_landmarks:
                landmarks = calculateLandmarks(frameRgb, hand_landmarks)
                processed_landmarks = processLandmarks(landmarks)

                signId, confidence = landmark_identifier(processed_landmarks)

                if confidence > 0.5:
                    try:
                        display_text(identifierLabels[signId - 1])
                    except IndexError:
                        print(f"Warning: Sign ID {signId} is out of range for identifierLabels.")
                        continue

def calculateLandmarks(image, landmarks):
    imageWidth, imageHeight = image.shape[1], image.shape[0]
    landmarkPoints = [
        [min(int(landmark.x * imageWidth), imageWidth - 1), min(int(landmark.y * imageHeight), imageHeight - 1)]
        for landmark in landmarks.landmark
    ]
    return landmarkPoints

def processLandmarks(landmark_list):
    landmarks = np.array(landmark_list)

    landmarks2 = landmarks - landmarks[0]
    flattenedLandmarks = landmarks2.ravel()

    maxValue = np.max(np.abs(flattenedLandmarks))
    if maxValue == 0: maxValue = 1

    landmarks2 = flattenedLandmarks / maxValue

    return landmarks2.tolist()

if __name__ == '__main__':
    initialise()
    main()
