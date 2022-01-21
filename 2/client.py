import socket
import pickle as pkl
import time

def train(weight_file, version):
    # just up the version!
    with open(weight_file, 'rb') as file:
        old_weights = pkl.load(file)
    #TRAIN
    new_weights = old_weights
    with open("weights_{:04d}.pkl".format(version+1),'wb') as file:
        pkl.dump(new_weights, file)
    
HEADERSIZE = 10

IP = "192.168.7.139"
PORT = 1234

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((IP, PORT))

# get weight and version
message_header = s.recv(HEADERSIZE)
message_length = int(message_header.decode('utf-8').strip())
version = s.recv(message_length).decode('utf-8')
version = int(version)
print('Getting weights version: {}'.format(version))

with open("weights_{:04d}.pkl".format(version),'wb') as file:
    message_header = s.recv(HEADERSIZE)
    message_length = int(message_header.decode('utf-8').strip())
    weights = s.recv(message_length)
    file.write(weights)
print('Done!')

print('training')
train('weights_{:04d}.pkl'.format(version), version)

print('Sending weights back to server')
filename = "weights_{:04d}.pkl".format(version+1)
with open(filename, 'rb') as file:
    sendfile = file.read()
sendfile = bytes(f"{len(sendfile):<{HEADERSIZE}}",'utf-8')+sendfile
s.send(sendfile)
print('Done!')