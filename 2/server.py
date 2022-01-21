import socket
import time
import pickle as pkl
import os

IP = "192.168.1.5"
PORT = 1234

weight_file = [i for i in os.listdir() if (i[-3:]=='pkl' and i[:7]=='weights')][-1]
weight_version = int(weight_file[8:12])

HEADERSIZE = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((IP, PORT))
s.listen(5)

clientsocket, address = s.accept()
print(f"Connection from {address} has been established.")
print('Sending weights: {}'.format(weight_file))
msg = str(weight_version).encode('utf-8')
msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8')+msg
clientsocket.send(msg)

filename = weight_file
with open(filename, 'rb') as file:
    sendfile = file.read()
sendfile = bytes(f"{len(sendfile):<{HEADERSIZE}}",'utf-8')+sendfile
clientsocket.send(sendfile)
print('Done!')

print('getting new weights')
with open("weights_{:04d}.pkl".format(weight_version+1),'wb') as file:
    message_header = clientsocket.recv(HEADERSIZE)
    message_length = int(message_header.decode('utf-8').strip())
    weights = clientsocket.recv(message_length)
    file.write(weights)
print('Done!')