from tkinter import S
from utils import *
from face_recognition import face_recognition_network
from PIL import Image
import numpy as np
import os
import cv2
import time


d = update_known_embeddings('update')
nn = face_recognition_network()
nn.load_weights('latest')
all_names = []
for p in d:
    if p['name'] not in all_names:
        all_names.append(p['name'])

print('\n\n')
print('---------------------Running Attendance System for webcam--------------------')


# initialize the camera
cv2.namedWindow("preview")
cam = cv2.VideoCapture(0)   # 0 -> index of camera

if cam.isOpened(): # try to get the first frame
    s, img = cam.read()
else:
    s = False

while s:
    print('- Time: {}'.format(time.strftime("%H:%M:%S", time.localtime())))
    s, img = cam.read()
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = test_model_with_image(img, nn, d)
    attendees_list = []
    absent_list = all_names.copy()
    if len(r)!=0:
        for face in r:
            x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
            out_img = cv2.rectangle(out_img, (x1, y1), (x2, y2), (36,255,12), 1)
            out_img = cv2.putText(out_img, face['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            if face['name'] != 'Unknown':
                attendees_list.append(face['name'])
                absent_list.remove(face['name'])
    print('  -- Atendees:')
    for a in attendees_list:
        print('    - {}'.format(a))
    print('  -- Absent:')
    for a in absent_list:
        print('    - {}'.format(a))
    print('')
    img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("preview", img)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cam.release()
cv2.destroyWindow("preview")
