import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
from utils import *
from face_recognition import face_recognition_network

def train_model(epochs=20, weights='latest'):
    database = update_known_embeddings('update')
    X_train, X_val, y_train, y_val = make_train_dataset(database)
    print('---------------------Training Face recognition Model---------------------')
    nn = face_recognition_network()
    nn.load_weights(weights)
    print('- Training Model')
    h = nn.fit(X_train, X_val, y_train, y_val, epochs)
    print('- Training Done. Results after {} epochs:'.format(epochs))
    print('  -- Train Loss: {:.4f}'.format(h['loss'][-1]))
    print('  -- Val Loss: {:.4f}'.format(h['val_loss'][-1]))
    print('  -- Train Acc: {:.4f}'.format(h['accuracy'][-1]))
    print('  -- Val Acc: {:.4f}'.format(h['val_accuracy'][-1]))
    nn.save_weights()
    print('_________________________________________________________________________')
    print('')
    return nn

