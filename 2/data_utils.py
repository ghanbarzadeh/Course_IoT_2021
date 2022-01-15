import random
import numpy as np
import os
from matplotlib.image import imread
import cv2


def create_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    random.seed(2021)
    pairImages = []
    pairLabels = []
   
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(y_val))
    classes=np.unique(y_val)
    idx = [np.where(y_val == classes[i]) for i in range(0, numClasses)]
    
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current iteration
        currentImage = images[idxA]
        label = labels[idxA]
        
        # randomly pick an image that belongs to the *same* class
        # label
        posId = random.choice(list(np.where(labels == label)))
        posIdx =random.choice(posId)
        posImage = images[posIdx]
        
        # prepare a positive pair and update the images and labels
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negId = random.choice(list(np.where(labels != label)))         
        negIdx =random.choice(negId)
        negImage = images[negIdx]
        
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
   
    return (np.array(pairImages), np.array(pairLabels))

def resize_dataset_images(dataset_path, dim=(154, 154)):
    for folder in os.listdir(dataset_path):
        for image_path in os.listdir(os.path.join(dataset_path, folder)):
            image = imread(os.path.join(dataset_path, folder, image_path))
            img_shape = image.shape
            if img_shape[0]!=dim[0] or img_shape[1]!=dim[1] or True:
                image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                os.remove(os.path.join(dataset_path, folder, image_path))
                cv2.imwrite(os.path.join(dataset_path, folder, image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
