import socket
import pickle as pkl
import time
from train_head_model import train_model
    
HEADERSIZE = 10

IP = "192.168.0.1"
PORT = 1234

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((IP, PORT))

# get weight and version
message_header = s.recv(HEADERSIZE)
message_length = int(message_header.decode('utf-8').strip())
version = s.recv(message_length).decode('utf-8')
version = int(version)
print('Getting weights version: {}'.format(version))

with open("weights_{:04d}.h5".format(version),'wb') as file:
    message_header = s.recv(HEADERSIZE)
    message_length = int(message_header.decode('utf-8').strip())
    weights = s.recv(message_length)
    file.write(weights)
print('Done!')

print('training')
train_model('weights_{:04d}.h5'.format(version), version)

print('Sending weights back to server')
filename = "weights_{:04d}.h5".format(version+1)
with open(filename, 'rb') as file:
    sendfile = file.read()
sendfile = bytes(f"{len(sendfile):<{HEADERSIZE}}",'utf-8')+sendfile
s.send(sendfile)
print('Done!')