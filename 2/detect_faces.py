# This function takes an image array and returns a list containing faces x,y,w,h

import cv2 
import numpy as np

def detect_faces(image_array):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
    return faces