import asyncio
import io
import glob
import os
import sys
import time
import json
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition

import config

subscription_key = config.KEY
assert subscription_key

face_api_url = config.ENDPOINT
assert face_api_url

stamp = round(time.time())


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
    response = requests.get(single_face_image_url)
    img = Image.open(BytesIO(response.content))

    fnt = ImageFont.truetype('../assets/Roboto-Regular.ttf', 50)

    # For each face returned use the face rectangle and draw a red box.
    print('Drawing rectangle around face... see popup for results.')
    draw = ImageDraw.Draw(img)
    for face in detected_faces:
        draw.rectangle(getRectangle(face), outline='red')
        # write the predominant emotion over the bounding box
        draw.text(getCoordsForText(face), str(getMainEmotion(face)), fill=(255, 255, 255, 255), font=fnt)

    img.show()
    img.save(os.getcwd() + "/test-images/" + single_image_name + "_modified.png", "PNG")

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

    # print(emotionList)
    print('Predominant emotion is ' + str(emotionList[0]))
    return emotionList[0]

# Create an authenticated FaceClient.
face_client = FaceClient(face_api_url, CognitiveServicesCredentials(subscription_key))

# Detect a face in a random image that contains a single face
single_face_image_url = 'https://fakeface.rest/face/view'
single_image_name = str(stamp)

attrs = ["age", "gender", "headPose", "smile", "facialHair", "glasses", "emotion", "hair", "makeup", "occlusion", "accessories", "blur", "exposure", "noise"]

detected_faces = face_client.face.detect_with_url(url=single_face_image_url,
                                                  detection_model='detection_01',
                                                  return_face_attributes=attrs)
if not detected_faces:
    raise Exception('No face detected from image {}'.format(single_image_name))

# Display the detected face ID in the first single-face image.
# Face IDs are used for comparison to faces (their IDs) detected in other images.
print('Detected face ID from', single_image_name, ':')
for face in detected_faces:
    print(face.face_id)
print()

drawFaceRectangles()