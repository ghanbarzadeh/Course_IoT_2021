from utils import *
from face_recognition import face_recognition_network
from PIL import Image
import numpy as np
import os
import cv2
import glob

d = update_known_embeddings('update')
nn = face_recognition_network()
nn.load_weights('latest')

test_images = glob.glob('test_images/*.jpg')
OUTPUT_FOLDER= 'test_results'

print('\n\n')
print('-------------------------Testing Images from folder-------------------------')

print('- Found {} Images in test folder'.format(len(test_images)))
print('- Testing Images')

for image_path in test_images:
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_array = np.asarray(image)
    r = test_model_with_image(image_array, nn, d)
    for face in r:
        x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
        image_array = cv2.rectangle(image_array, (x1, y1), (x2, y2), (36,255,12), 1)
        image_array = cv2.putText(image_array, face['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    image_name = os.path.split(image_path)[-1]
    Image.fromarray(image_array).save(os.path.join(OUTPUT_FOLDER, image_name))
    print('  -- Wrote image to file: {}'.format(os.path.join(OUTPUT_FOLDER, image_name)))

print('_________________________________________________________________________')
print('')

