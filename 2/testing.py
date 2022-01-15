from re import S
from face_recognition import face_recognition_network
from data_utils import resize_dataset_images
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from detect_faces import detect_faces
import cv2

im1 = imread(r"datasets\lfw_images\15-93.jpeg")
im2 = imread(r"datasets\lfw_images\15-40.jpeg")
im3 = imread(r"datasets\lfw_images\14-62.jpeg")


x = np.array([[im1, im2],[im1,im3]])

nn = face_recognition_network()
nn.load_weights("weights_0.pkl")

print(nn.predict(x))