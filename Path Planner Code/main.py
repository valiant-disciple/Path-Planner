# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:23:33 2023

@author: Arya
"""

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

"""Call your functions for the motion of the car here"
