# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:23:33 2023

@author: Arya
"""

import numpy as np
import cv2
from PIL import Image
import Algorithm
import ObjectDetection 

data = ObjectDetection.Starter()
print(data.dtype)

# image = Image.open('pixel_map.png')
# data = np.array(image)

myAlgo = Algorithm.A_Star_Algo(data)
protocol = myAlgo.algo_exec()

'''Main should recieve path_points and then we open the pixel map here'''
# print(myAlgo.closed_list)
# print(myAlgo.open_list)