from re import S
from face_recognition import face_recognition_network
from data_utils import resize_dataset_images
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from detect_faces import detect_faces
import cv2

im1 = imread(r"people\Armin Ghanbarzadeh\1.jpg") / 255
im2 = imread(r"people\Armin Ghanbarzadeh\2.jpg") / 255
im3 = imread(r"people\Mohammad Hoseinzadeh\1.jpg") / 255


x = np.array([[im1, im2],[im1,im3]])

nn = face_recognition_network()
nn.load_weights("weights_0.pkl")

print(nn.predict(x))