from utils import *
from train_head_model import train_model
import numpy as np
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

# train_model()
a = {'Armin_Ghanbarzadeh': 0.9686954220136007, 'ben_afflek': 4.011020190185970731, 'elton_john': 0.059070247979391186, 'jerry_seinfeld': 0.010657280683517456, 'madonna': 0.01256458914798239, 'mindy_kaling': 0.08615383391196911, 'Mohammad_Hoseinzadeh': 0.04373494478372427, 'navid_faraji': 0.051495937837494746, 'Reza_Behbahani': 0.013368189334869385}
b = np.argmax(list(a.values()))
print(b)
print(list(a.keys())[b])