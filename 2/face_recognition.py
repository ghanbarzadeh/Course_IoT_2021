import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pickle as pkl
import glob
from keras import backend as K
import cv2
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np

def euclidean_distance(vectors):
        featsA, featsB = vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,keepdims=True)
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))


class face_recognition_network:

    def __init__(self, verbose=True):
        self.model = self.make_model()
        self.verbose = verbose
        if self.verbose:
            print('* Model created and initialized with random weights')
        self.weights_version = -1
        self.WEIGHTS_PATH = 'models'
        # self.model.summary() 

    def make_model(self):
        input_1 = tf.keras.layers.Input(shape=(128,))
        input_2 = tf.keras.layers.Input(shape=(128,))
        distance = tf.keras.layers.Lambda(euclidean_distance)([input_1, input_2])
        x = Dense(100, activation='relu')(distance)
        output = Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
        return model


    def fit(self, x_train, x_val, y_train, y_val, epochs):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
        history = self.model.fit([x_train[:,0], x_train[:,1]], y_train[:], epochs=epochs, 
                                    validation_data=[[x_val[:,0],x_val[:,1]], y_val[:]], verbose=0)
        self.weights_version += 1
        return history.history
    

    def predict(self, x_test):
        x_test = [x_test[:, 0],x_test[:, 1]]
        y_test = self.model.predict(x_test)
        # y_test = np.where(y_test, y_test>0.7, 1)
        return y_test

    
    def load_weights(self, weights=None):
        if weights=='latest':
            weight_files = glob.glob(os.path.join(self.WEIGHTS_PATH, '*.h5'))
            if len(weight_files)==0:
                weights = None
                self.weights_version = -1
            else:
                weight_files.sort()
                weights = os.path.split(weight_files[-1])[-1]
        if weights==None:
            return None
        self.weights_version = int(weights[8:12])
        self.model.load_weights(os.path.join(self.WEIGHTS_PATH, weights))
        if self.verbose:
            print('* Loaded weights version {} from file {}'.format(self.weights_version, os.path.join(self.WEIGHTS_PATH, weights)))
        return self.weights_version

    
    def load_weights_from_base(self, weights_path):
        weights_file = os.path.split(weights_path)[-1]
        self.weights_version = int(weights_file[17:21])
        self.model.load_weights(weights_path)
        if self.verbose:
            print('* Loaded weights version {} from file {}'.format(self.weights_version, weights_path))
        return self.weights_version
    

    def save_weights(self, weights_file=None):
        if weights_file:
            self.model.save_weights(weights_file)
        else:
            weights_file = os.path.join(self.WEIGHTS_PATH, 'weights_{:04d}.h5'.format(self.weights_version))
            self.model.save_weights(weights_file)
        if self.verbose:
            print('* Saved weights version {} to file {}'.format(self.weights_version, os.path.join(self.WEIGHTS_PATH, weights_file)))