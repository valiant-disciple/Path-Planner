import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostbyname(socket.gethostname()), 1234))
s.listen(5)
txt=socket.gethostbyname(socket.gethostname())
while True:
   
    clientsocket, address = s.accept()
    print(f"Connection established.")
    #clientsocket.send(bytes("TEST MESSAGE","utf-8"))
    print(txt)
