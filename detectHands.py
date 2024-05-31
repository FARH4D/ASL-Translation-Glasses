import csv
import os
import cv2
import mediapipe as mp
import numpy as np

def identifyLandmarks(classId, image):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            imageWidth, imageHeight = image.shape[1], image.shape[0]

            landmarkPoints = [
                [min(int(landmark.x * imageWidth), imageWidth - 1), min(int(landmark.y * imageHeight), imageHeight - 1)]
                for landmark in hand_landmarks.landmark
            ]
            processLandmarks(classId, landmarkPoints)
    return True

def processLandmarks(classId, landmark_list):
    landmarks = np.array(landmark_list)

    landmarks2 = landmarks - landmarks[0]

    flattenedLandmarks = landmarks2.ravel()

    maxValue = np.max(np.abs(flattenedLandmarks))
    if maxValue == 0: maxValue = 1
    scaledLandmarks = flattenedLandmarks / maxValue

    final_landmark_list = scaledLandmarks.tolist()
    final_landmark_list.insert(0, classId)

    writeToCsv(final_landmark_list)
    
def writeToCsv(data):
    csv_file = 'model/data/landmarks.csv'
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def main():
    ## REMEMBER TO CHANGE THESE TWO VALUES EVERYTIME YOU WANT TO ADD A NEW CLASS.
    ## REMEMBER ID STARTS AT 0 !!!!!!!!
    ## ALSO REMEMBER TO MANUALLY CREATE EACH FOLDER FOR SIGN
    sign_prefix = "Moon."
    classId = 27

    images_path = 'images'

    image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

    for image_file in image_files:
        if image_file.startswith(sign_prefix) and image_file.endswith(('.jpg')):
            image_path = os.path.join(images_path, image_file)
            image = cv2.imread(image_path)

            hand_detected = identifyLandmarks(classId, image)
            
            if hand_detected:
                print("Hand landmarks detected in " + image_file)

if __name__ == "__main__":
    main()