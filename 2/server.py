import socket
import os
import glob
from utils import *


IP = "127.0.0.1"
PORT = 1234
HEADERSIZE = 10

NUMBER_OF_CLIENTS = 2
WEIGHTS_FOLDER = 'server_weights'

weight_files = glob.glob(os.path.join(WEIGHTS_FOLDER, '*.h5'))
weight_files.sort()
weight_file = os.path.split(weight_files[-1])[-1]
weight_version = int(weight_file[8:12])

print('---------------------Starting Server---------------------')
print('IP: {}, Port: {}'.format(IP, PORT))
print('- Waiting for {} client(s)'.format(NUMBER_OF_CLIENTS))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((IP, PORT))
s.listen(5)


clients = []
for i in range(NUMBER_OF_CLIENTS):
    clientsocket, address = s.accept()
    clients.append((clientsocket, address))
    print(f"  -- Connection from {address} has been established.")
    print('    -- Sending weights version: {} to client# {}'.format(weight_version, i))
    msg = str(weight_version).encode('utf-8')
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8')+msg
    clientsocket.send(msg)

    filename = os.path.join(WEIGHTS_FOLDER, weight_file)
    with open(filename, 'rb') as file:
        sendfile = file.read()
    sendfile = bytes(f"{len(sendfile):<{HEADERSIZE}}",'utf-8')+sendfile
    clientsocket.send(sendfile)
    print('    -- Done')
    print('')

print('')
print('- Waiting for new weights')
for i in range(NUMBER_OF_CLIENTS):
    clientsocket, address = clients[i]
    new_weights_filename = os.path.join(WEIGHTS_FOLDER, "client_{}_weights_{:04d}.h5".format(i, weight_version+1))
    with open(new_weights_filename,'wb') as file:
        message_header = clientsocket.recv(HEADERSIZE)
        message_length = int(message_header.decode('utf-8').strip())
        weights = clientsocket.recv(message_length)
        file.write(weights)
    print('    -- Received weights version: {} from client# {}'.format(weight_version+1, i))
    print('     --- Saved to file {}'.format(new_weights_filename))
    print('')

print('')
print('- Averaging the weights from clients')
print('')

all_weights = os.listdir(WEIGHTS_FOLDER)
client_weights = []
for file in all_weights:
    if file[0] == 'c':
        client_weights.append(file)
client_weights = [os.path.join(WEIGHTS_FOLDER, cw) for cw in client_weights]

save_path = os.path.join(WEIGHTS_FOLDER, "weights_{:04d}.h5".format(weight_version+1))
new_weights = average_weights(client_weights, save_path)

for file in client_weights:
    os.remove(file)

print('- Sending new weights version {} to clients'.format(weight_version+1))

weights_file = os.path.join(WEIGHTS_FOLDER, "weights_{:04d}.h5".format(weight_version+1))

for i in range(NUMBER_OF_CLIENTS):
    clientsocket, address = clients[i]
    print('    -- Sending weights version: {} to client# {}'.format(weight_version+1, i))
    with open(weights_file, 'rb') as file:
        sendfile = file.read()
    sendfile = bytes(f"{len(sendfile):<{HEADERSIZE}}",'utf-8')+sendfile
    clientsocket.send(sendfile)
    print('    -- Done')
    print('')

