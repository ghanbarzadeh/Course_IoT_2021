from re import S
from face_recognition import face_recognition_network
from data_utils import resize_dataset_images
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from detect_faces import detect_faces
import cv2

nn = face_recognition_network()
nn.load_weights("weights_0.pkl")

test_images_path = "test"
test_images = os.listdir(test_images_path)

people_database_path = "people"
resize_dataset_images(people_database_path)
people = os.listdir(people_database_path)

for test_image in test_images:
    image_path = os.path.join(test_images_path, test_image)
    im = imread(image_path)
    faces_coordinate = detect_faces(im)

    present_people = [] # Empty list to populate with people
    for face_coordinate in faces_coordinate:
        x, y, w, h = face_coordinate
        face_image = im[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (154, 154), interpolation = cv2.INTER_AREA)
        
        s = []
        for p in people:
            p_images = []
            for p_image in os.listdir(os.path.join(people_database_path, p)):
                p_images.append(imread(os.path.join(people_database_path, p, p_image)))
            dual_images = np.array([[face_image, i] for i in p_images])
            s.append(np.mean(nn.predict(dual_images)))
        s = np.array(s)
        print(s)
        if np.max(s) > 0.4:
            present_people.append(people[np.argmax(s)])
    print(present_people)
