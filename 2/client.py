import socket
import pickle as pkl
import time
from train_head_model import train_model
import os
    
HEADERSIZE = 10
WEIGHTS_FOLDER = 'models'

IP = "127.0.0.1"
PORT = 1234

print('---------------------Starting Client---------------------')
print('- Connecting to server on IP: {}, Port: {}'.format(IP, PORT))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((IP, PORT))

message_header = s.recv(HEADERSIZE)
message_length = int(message_header.decode('utf-8').strip())
version = s.recv(message_length).decode('utf-8')
version = int(version)
print('- Getting weights version: {}'.format(version))

with open(os.path.join(WEIGHTS_FOLDER, "weights_{:04d}.h5".format(version)),'wb') as file:
    message_header = s.recv(HEADERSIZE)
    message_length = int(message_header.decode('utf-8').strip())
    weights = s.recv(message_length)
    file.write(weights)
print(' - Weights received')
print('  -- Saving weights to file: {}'.format(os.path.join(WEIGHTS_FOLDER, "weights_{:04d}.h5".format(version))))

print(' - Training with latest weights')
train_model(weights="weights_{:04d}.h5".format(version))

print(' - Sending weights version {} back to server'.format(version+1))
filename = os.path.join(WEIGHTS_FOLDER, "weights_{:04d}.h5".format(version+1))
with open(filename, 'rb') as file:
    sendfile = file.read()
sendfile = bytes(f"{len(sendfile):<{HEADERSIZE}}",'utf-8')+sendfile
s.send(sendfile)
print(' - Weights sent')

print('- Getting new weights version: {}'.format(version+1))

with open(os.path.join(WEIGHTS_FOLDER, "weights_{:04d}.h5".format(version+1)),'wb') as file:
    message_header = s.recv(HEADERSIZE)
    message_length = int(message_header.decode('utf-8').strip())
    weights = s.recv(message_length)
    file.write(weights)
print(' - New weights received')
print('  -- Saving weights to file: {}'.format(os.path.join(WEIGHTS_FOLDER, "weights_{:04d}.h5".format(version+1))))