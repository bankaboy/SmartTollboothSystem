
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import imutils
from darkflow.net.build import TFNet
import random
import time
from flask import Flask, render_template, request, flash, request, redirect, url_for, jsonify
import json
import requests
from statistics import mode

# used to detect plate using yolo model
options = {"pbLoad": "Plate_recognition_weights/yolo-plate.pb", "metaLoad": "Plate_recognition_weights/yolo-plate.meta", "gpu": 0.9}
yoloPlate = TFNet(options)

# used to detect characters on number plate
options = {"pbLoad": "Character_recognition_weights/yolo-character.pb", "metaLoad": "Character_recognition_weights/yolo-character.meta", "gpu":0.9}
yoloCharacter = TFNet(options)

characterRecognition = tf.keras.models.load_model('character_recognition.h5')

# function that returns the cropped  detection with the highest confidence (last in confidence sorted list)
# draws rectangle around the highest confidence of license plate detecttions given to it
def firstCrop(img, predictions):
    predictions.sort(key=lambda x: x.get('confidence'))
    xtop = predictions[-1].get('topleft').get('x')
    ytop = predictions[-1].get('topleft').get('y')
    xbottom = predictions[-1].get('bottomright').get('x')
    ybottom = predictions[-1].get('bottomright').get('y')
    firstCrop = img[ytop:ybottom, xtop:xbottom]
    cv2.rectangle(img,(xtop,ytop),(xbottom,ybottom),(0,255,0),3)
    return firstCrop
    
# applies contour function on top of the image
# used on top of the cropped out license plate
def secondCrop(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        secondCrop = img[y:y+h,x:x+w]
    else: 
        secondCrop = img
    return secondCrop

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

# find the characters in the image and return as a string
# works on top of the contoured license plate image
def opencvReadPlate(img):
    charList=[]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
    edges = auto_canny(thresh_inv)
    ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = img.shape[0]*img.shape[1]

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w*h
        non_max_sup = roi_area/img_area

        if((non_max_sup >= 0.015) and (non_max_sup < 0.09)):
            if ((h>1.2*w) and (3*w>=h)):
                char = img[y:y+h,x:x+w]
                charList.append(cnnCharRecognition(char))
                cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
    # cv2.imshow('OpenCV character segmentation',img)
    licensePlate="".join(charList)
    return licensePlate

# used to detect characters using keras model on the license plate
def cnnCharRecognition(img):
    dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
    11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
    21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
    30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}

    blackAndWhiteChar=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackAndWhiteChar = cv2.resize(blackAndWhiteChar,(75,100))
    image = blackAndWhiteChar.reshape((1, 100,75, 1))
    image = image / 255.0
    new_predictions = characterRecognition.predict(image)
    char = np.argmax(new_predictions)
    return dictionary[char]

def main(booth_video_file):
    cap = cv2.VideoCapture(booth_video_file)
    counter=0
    car_dict = {'Car License':'AAAAAA','Timestamp':100000000,'Booth ID':1}
    licensePlates = []
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        h, w, l = frame.shape
        frame = imutils.rotate(frame, 270)

        if counter%6 == 0:
            try:
                booth_number = random.randint(1,5)
                predictions = yoloPlate.return_predict(frame)
                firstCropImg = firstCrop(frame, predictions)
                # cv2.imshow('First crop plate',firstCropImg)
                secondCropImg = secondCrop(firstCropImg)
                # cv2.imshow('Second crop plate',secondCropImg)
                secondCropImgCopy = secondCropImg.copy()
                licensePlates.append(opencvReadPlate(secondCropImg))
            except Exception as e:
                # pass
                print('EXCEPTION: ', e)
        if counter%36 == 0:
            try:
                # print(licensePlates)
                licensePlate = mode(licensePlates)
                # create a single record for the vehicle
                ts =  time.time()
                car_dict['Timestamp'] = ts
                car_dict['Car License'] = licensePlate
                car_dict['Booth ID'] = booth_number
                # car_record = json.dumps(car_dict)
                # print(car_record)
                request = requests.post("http://127.0.0.1:8080/tollbooths", data=car_dict)
            except Exception as e:
                # pass
                print('EXCEPTION: ', e)
            licensePlates.clear()
            # print(licensePlates)

        counter+=1

    cap.release()
    cv2.destroyAllWindows()

main('./test_videos/vid1.MOV')