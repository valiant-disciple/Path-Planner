# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:23:33 2023
@author: Arya
"""
import socket

#import numpy as np
#import cv2
#from PIL import Image
import Algorithm
import ObjectDetection 

data = ObjectDetection.Starter() #Object Detection starts here, returns pixel map of img.
# print(data.dtype)

# image = Image.open('pixel_map.png')
# data = np.array(image)

myAlgo = Algorithm.A_Star_Algo(data) #Runs the path-finding algorithm.
protocol = myAlgo.algo_exec() #Text Protocol for motion.



s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostbyname(socket.gethostname()), 1234))
s.listen(5)
txt=socket.gethostbyname(socket.gethostname())
while True:
   
    clientsocket, address = s.accept()
    print(f"Connection established.")
    clientsocket.send(bytes(protocol,"utf-8"))
    print(txt)
