# Basic image manipulation scripts to preprocess the new images

import cv2
from matplotlib.pyplot import imread
import os
from PIL import Image
import glob

PATH = r'people\Armin_Ghanbarzadeh'

for filename in glob.glob(PATH + '/*.jpg'): #assuming gif
    im = imread(filename)
    width = int(im.shape[1] * 8 / 100)
    height = int(im.shape[0] * 8 / 100)
    dim = (width, height)
    im = cv2.resize(im, dim)
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(filename, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    print('wrote image to: {}'.format(filename))
