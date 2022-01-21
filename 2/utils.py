import os
from re import L
from telnetlib import X3PAD
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import pickle as pkl
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split


def extract_faces(image, required_size=(160, 160)):
    detector = MTCNN()
    results = detector.detect_faces(image)
    faces = []
    for i in range(len(results)):
        x1, y1, w, h = results[i]['box']
        x1, y1 = np.abs(x1), np.abs(y1)
        x2, y2 = x1 + w, y1 + h
        face = image[y1:y2, x1:x2]
        face = Image.fromarray(face)
        face = face.resize(required_size)
        face = np.asarray(face)
        faces.append([face, (x1,y1), (x2,y2)])
    return faces


def get_embedding(faces_array):
    face_net = load_model("models/facenet")
    mean, std = faces_array.mean(), faces_array.std()
    samples = (faces_array - mean) / std
    embeddings = face_net.predict(samples)
    in_encoder = Normalizer(norm='l2')
    embeddings = in_encoder.transform(embeddings)
    return embeddings


def update_known_embeddings(embedding_file=None):
    print('-------------------------Updating Embedding File-------------------------')
    PEOPLE_FOLDER_PATH = r"people"
    if embedding_file=='update':
        if 'known_people_database.pkl' in os.listdir(r"datasets"):
            embedding_file = 'datasets/known_people_database.pkl'
        else:
            embedding_file = None
    if embedding_file:
        print('- Using embedding file: {}'.format(embedding_file))
        with open(embedding_file, 'rb') as f:
            old_database = pkl.load(f)
        print('- Total embeddings found in file: {}'.format(len(old_database)))
        old_paths = [old_database[i]['path'] for i in range(len(old_database))]
        database = []
        temp_database = []
        images_array = []
        for name in os.listdir(PEOPLE_FOLDER_PATH):
            for image_path in os.listdir(os.path.join(PEOPLE_FOLDER_PATH, name)):
                if os.path.join(PEOPLE_FOLDER_PATH, name, image_path) in old_paths:
                    database.append(old_database[old_paths.index(os.path.join(PEOPLE_FOLDER_PATH, name, image_path))])
                else:
                    dic = {}
                    dic['path'] = os.path.join(PEOPLE_FOLDER_PATH, name, image_path)
                    dic['name'] = name
                    dic['embedding'] = 0
                    temp_database.append(dic)
                    images_array.append(np.asarray(Image.open(dic['path']).convert('RGB')))
        faces = []
        if len(temp_database)!=0:
            for image in images_array:
                face = extract_faces(image)
                face = face[0][0]
                faces.append(face)
            faces_array = np.asarray(faces).astype('float32')
            embeddings = get_embedding(faces_array)
            for i in range(len(temp_database)):
                temp_database[i]['embedding'] = embeddings[i]
        print('- Finished updating dataset')
        print('  -- New images: {}'.format(len(temp_database)))
        print('  -- Old images: {}'.format(len(database)))
        print('  -- Removed images: {}'.format(len(old_database)-len(database)))
        database.extend(temp_database)
        with open('datasets/known_people_database.pkl','wb') as f:
            pkl.dump(database, f)
        print('_________________________________________________________________________')
        print('')
        return database
    else:
        print('- No file specified, creating new embedding file')
        database = []
        images_array = []
        for name in os.listdir(PEOPLE_FOLDER_PATH):
            for image_path in os.listdir(os.path.join(PEOPLE_FOLDER_PATH, name)):
                dic = {}
                dic['path'] = os.path.join(PEOPLE_FOLDER_PATH, name, image_path)
                dic['name'] = name
                dic['embedding'] = 0
                database.append(dic)
                images_array.append(np.asarray(Image.open(dic['path']).convert('RGB')))
        faces = []
        for image in images_array:
            face = extract_faces(image)
            face = face[0][0]
            faces.append(face)
        faces_array = np.asarray(faces).astype('float32')
        embeddings = get_embedding(faces_array)
        for i in range(len(database)):
            database[i]['embedding'] = embeddings[i]
        with open('datasets/known_people_database.pkl','wb') as f:
            pkl.dump(database, f)
        print('- Finished making embedding file')
        print('  -- Total images found: {}'.format(len(database)))
        print('  -- Saved embedding file to: {}'.format('datasets/known_people_database.pkl'))
    print('_________________________________________________________________________')
    print('')
    return database
        
        
def make_train_dataset(database):
    print('-------------------------Making Training Dataset-------------------------')
    print('- Loading images')
    names = {}
    labels = {}
    l = 0
    for i in range(len(database)):
        name = database[i]['name']
        if name in names:
            names[name] += 1
        else:
            names[name] = 0 
            labels[name] = l
            l = l+1
    for name in names:
        print('  -- Loaded {} images of class: {}'.format(names[name], name))
    X = []
    y = []
    for i in range(len(database)):
        X.append(database[i]['embedding'])
        y.append(labels[database[i]['name']])
    X = np.asarray(X)
    y = np.asarray(y)
    trainX, testX, trainy, testy = train_test_split(X, y, train_size=0.85)
    print('- Creating Dataset for training')
    X_train = []
    y_train = []

    X_val = []
    y_val = []

    for i in range(trainX.shape[0]):
        c = trainX.shape[0] - i
        for j in range(c):
            X_train.append([trainX[i,:],trainX[j,:]])
            if trainy[i] == trainy[j]:
                y_train.append(1)
            else:
                y_train.append(0)

    for i in range(testX.shape[0]):
        c = testX.shape[0] - i
        for j in range(c):
            X_val.append([testX[i,:],testX[j,:]])
            if testy[i] == testy[j]:
                y_val.append(1)
            else:
                y_val.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    print('  -- Train dataset size: {}'.format(X_train.shape[0]))
    print('  -- Test dataset size: {}'.format(X_val.shape[0]))
    print('_________________________________________________________________________')
    print('')
    return X_train, X_val, y_train, y_val


def test_model_with_image(image_array, model, database):
    faces_data = extract_faces(image_array)
    if len(faces_data)==0:
        return None
    names_array = [d['name'] for d in database]
    names_count = {}
    for i in range(len(names_array)):
            if names_array[i] in names_count:
                names_count[names_array[i]] += 1
            else:
                names_count[names_array[i]] = 0
    res = []
    faces = []
    for face_data in faces_data:
        faces.append(face_data[0])
    faces = np.asarray(faces)
    face_embeddings = get_embedding(faces)
    for i in range(len(face_embeddings)):
        x_test = []
        for j in range(len(database)):
            x_test.append([face_embeddings[i], database[j]['embedding']])
        x_test = np.asarray(x_test)
        x_test = x_test.astype('float32')
        y_test = model.predict(x_test)
        r = {}
        for k in range(len(y_test)):
            score = y_test[k][0]
            if names_array[k] in r:
                r[names_array[k]] += score
            else:
                r[names_array[k]] = 0
        for name in r:
            r[name] = r[name]/names_count[name]
        if np.max(list(r.values()))<0.5:
            n = 'Unknown'
        else:
            n = list(r.keys())[np.argmax(list(r.values()))]
        res.append({'name':n,
                    'x1':faces_data[i][1][0],
                    'y1':faces_data[i][1][1],
                    'x2':faces_data[i][2][0],
                    'y2':faces_data[i][2][1]})
    return res
        
