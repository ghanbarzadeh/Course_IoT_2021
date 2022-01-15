import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pickle as pkl
from model_utils import euclidean_distance


class face_recognition_network:

    def __init__(self):
        self.model = self.make_model()
        # self.model.summary()


    def tf_siamese_nn(self, shape=(154, 154, 3), embedding=64):
        inputs = tf.keras.layers.Input(shape)

        x = tf.keras.layers.Conv2D(96, (11, 11), padding="same")(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Conv2D(256, (5, 5), padding="same")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Conv2D(384, (3, 3), padding="same")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x=tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.activations.relu(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        outputs=tf.keras.layers.Dense(embedding)(x)
        model = tf.keras.Model(inputs, outputs)
        
        return model
    

    def make_model(self, shape=(154, 154, 3)):
        img1 = tf.keras.layers.Input(shape)
        img2 =  tf.keras.layers.Input(shape)
        featureExtractor = self.tf_siamese_nn()
        featsA = featureExtractor(img1)
        featsB = featureExtractor(img2)

        distance = tf.keras.layers.Lambda(euclidean_distance)([featsA, featsB])
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)

        model = tf.keras.Model(inputs=[img1, img2], outputs=outputs)
        return model


    def fit(self, x_train, x_val, y_train, y_val, lr, epochs):
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr, decay_steps=48000)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=15, epochs=epochs)
        return history
    

    def predict(self, x_test):
        x_test = [x_test[:, 0], x_test[:, 1]]
        y_test = self.model.predict(x_test)
        return y_test

    
    def load_weights(self, weights_file):
        with open(weights_file, 'rb') as f:
            weights = pkl.load(f)
        self.model.set_weights(weights)
    

    def save_weights(self, weights_file):
        with open(weights_file, 'wb') as f:
            pkl.dump(self.model.get_weights(), f)