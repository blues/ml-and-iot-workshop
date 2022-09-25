import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO, StringIO
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition
import cv2

import config

subscription_key = config.KEY
assert subscription_key

face_api_url = config.ENDPOINT
assert face_api_url

stamp = round(time.time())
single_image_name = str(stamp)
file_path = f'{os.getcwd()}/test-images/{single_image_name}'

face_client = FaceClient(face_api_url, CognitiveServicesCredentials(subscription_key))

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)

def getMainEmotion(faceDictionary):
    # Get the emotion collection and sort in ascending order to get the top result
    emotions = faceDictionary.face_attributes.emotion
    emotionList = {
                    "anger": emotions.anger,
                    "contempt": emotions.contempt,
                    "disgust": emotions.disgust,
                    "fear": emotions.fear,
                    "happiness": emotions.happiness,
                    "neutral": emotions.neutral,
                    "sadness": emotions.sadness,
                    "surprise": emotions.surprise
                }

    emotionList = sorted(emotionList.items(), key=lambda item: item[1], reverse=True)

    print('Predominant emotion is ' + str(emotionList[0]))
    return emotionList[0]

def getCoordsForText(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    return (left, top - 50)

def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height

    return ((left, top), (right, bottom))

def drawFaceRectangles() :
    img = Image.open(f'{file_path}.png')
    fnt = ImageFont.truetype('../assets/Roboto-Regular.ttf', 50)

    # For each face returned use the face rectangle and draw a red box.
    print('Drawing rectangle around face... see popup for results.')
    draw = ImageDraw.Draw(img)
    for face in detected_faces:
        draw.rectangle(getRectangle(face), outline='red')
        draw.text(getCoordsForText(face), str(getMainEmotion(face)), fill=(255, 255, 255, 255), font=fnt)

    img.show()
    img.save(f'{file_path}_modified.png', 'PNG')

# Detect a face from an image captured by the webcam
check, frame = webcam.read()

while True:
    try:
        check, frame = webcam.read()
        print(check)  # prints true as long as the webcam is running
        print(frame)  # prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename=f'{file_path}.png', img=frame)

            attrs = ["age", "gender", "headPose", "smile", "facialHair", "glasses", "emotion", "hair", "makeup", "occlusion", "accessories", "blur", "exposure", "noise"]

            detected_faces = face_client.face.detect_with_stream(open(f'{file_path}.png', 'rb'),
                                                              detection_model='detection_01',
                                                              return_face_attributes=attrs)
            if not detected_faces:
                raise Exception('No face detected from image {}'.format(single_image_name))

            # Display the detected face ID in the first single-face image.
            # Face IDs are used for comparison to faces (their IDs) detected in other images.
            print('Detected face ID from', single_image_name, ':')
            for face in detected_faces:
                print (face.face_id)
                print(f'Blur: {face.face_attributes.blur}')
                print(f'Emotion: {face.face_attributes.emotion}')
                print(f'Exposure: {face.face_attributes.exposure}')
                print(f'Head Pose: {face.face_attributes.head_pose}')
                print(f'Glasses: {face.face_attributes.glasses}')
                print(f'Facial Hair: {face.face_attributes.facial_hair}')
            print()

            drawFaceRectangles()

            print("Turning off camera.")
            webcam.release()
            cv2.destroyAllWindows()
            break

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        cv2.destroyAllWindows()
        break
