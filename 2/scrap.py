from utils import *
from train_head_model import train_model
import numpy as np
import h5py
from face_recognition import face_recognition_network
import time

# from train_head_model import train_model

# detector = MTCNN()

# # print('ok?')

# # model = load_model(r'models\facenet')

# img1 = r"people\ben_afflek\httpabsolumentgratuitfreefrimagesbenaffleckjpg.jpg"
# img2 = r"people\Armin_Ghanbarzadeh\IMG_20220119_173450.jpg"

# d = update_known_embeddings('datasets\known_people_database.pkl')
# make_train_dataset(d)
# d = update_known_embeddings()
# with open('datasets\known_people_database.pkl', 'rb') as f:
#     d = pkl.load(f)

# train_model(weights='latest')

# print(d)

# a = ['w001.pkl', 'w000.pkl', 'w003.pkl']
# a.sort()
# print(a)
# s = 'weights_9988.pkl'
# print(int(s[8:12]))

# # train_model()
# a = {'Armin_Ghanbarzadeh': 0.9686954220136007, 'ben_afflek': 4.011020190185970731, 'elton_john': 0.059070247979391186, 'jerry_seinfeld': 0.010657280683517456, 'madonna': 0.01256458914798239, 'mindy_kaling': 0.08615383391196911, 'Mohammad_Hoseinzadeh': 0.04373494478372427, 'navid_faraji': 0.051495937837494746, 'Reza_Behbahani': 0.013368189334869385}
# b = np.argmax(list(a.values()))
# print(b)
# print(list(a.keys())[b])

# hf = h5py.File('server_weights\weights_0001.h5', 'r')
# print(hf.keys())

# # print(hf)
# n1 = hf['dense_1']['dense_1']['bias:0']
# n1 = np.array(n1)
# print(n1.shape)

# nn = face_recognition_network(verbose=False)

# nn.load_weights('weights_0000.h5')
# a = nn.model.get_weights()

# nn.load_weights('weights_0001.h5')
# b = nn.model.get_weights()

# weights = [a, b]

# layer_weights = []

# for i in range(len(a)):
#     temp = []
#     for j in range(len(weights)):
#         temp.append(weights[j][i])
#     layer_weights.append(temp)
# print(layer_weights[-1])
# new_weights = []

# for i in range(len(a)):
#     temp = np.zeros(layer_weights[i][0].shape)
#     for j in range(len(weights)):
#         temp = temp + layer_weights[i][j]
#     temp = temp / len(weights)
#     new_weights.append(temp)
# print(new_weights[-1])
# a = ['models\weights_0000.h5', 'models\weights_0001.h5']
# print(average_weights(a)[1])


# from train_head_model import train_model

# train_model()

print(time.strftime("%H:%M:%S", time.localtime()))