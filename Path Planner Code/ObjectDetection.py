# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:24:52 2023

@author: Arya
"""
import cv2
import numpy as np

modelFile = "frozen_inference_graph.pb"
configFile = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
# classFile = "coco_class_labels.txt"
# if not os.path.isdir('models'):
#     os.mkdir("models")
#     if not os.path.isfile(modelFile):
#         os.chdir("models")
#         # Download the tensorflow Model
#         urllib.request.urlretrieve('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz', 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
        
#         # Uncompress the file
#         #!tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
#         tarfile.open(name = 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz', mode = 'r')
            
#         # Delete the tar.gz file
#         os.remove('ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
            
#         #Come back to the previous directory
#         os.chdir("..")
        
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

def ObjectDetector(net, img):
    tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    # Input image
    img = cv2.imread('img.png')
    rows, cols, channels = img.shape

    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size = (280, 280), swapRB=True, crop=False))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()
    return networkOutput

# def PrintObjectLabels(img, text, x, y):
#     textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#     dim = textSize[0]
#     baseline = textSize[1]
    
#     cv2.rectangle(img, (x, y-dim[1]-baseline), (x+dim[0], y+baseline, (0,0,0), cv2.FILLED)
#     cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
def DisplayObjects(img, objects, threshold):
    rows, cols, channels = img.shape
    final_map = np.ones(img.shape, dtype = 'uint8') * 255
    #print(final_map)
    
    for detection in objects[0,0]:

        score = float(detection[2])
        if score > 0.1:

            #Adjusting normalised coords back to usual coords
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            #Draw rectangle around detected objects
            cv2.rectangle(img, (int(left)-25, int(top)-25), (int(right)+25, int(bottom)+25), (0, 0, 255), thickness = 4)
            cv2.rectangle(final_map, (int(left)-25, int(top)-25), (int(right)+25, int(bottom)+25), (0, 0, 0), thickness = -1)
            cv2.rectangle(final_map, (0, 0), (cols, rows), (0, 0, 0), thickness = 2)
            final_map[2,2] = [255, 75, 15]
            final_map[rows - 3, cols - 3] = [255, 75, 15]
    
    cv2.imwrite('pixel_map2.png', final_map)
    #print(final_map)
    #print(Encoder(final_map))
    return final_map
    #cv2.imwrite('return_map.png', retu)
            
    # cv2.imshow('hi', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imshow('hi', final_map)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
def Encoder(final_map):
    rows , cols, channels = final_map.shape
    return_map = np.ones((rows, cols), 'uint8')
    #count=0
    for x in range(rows):
        for y in range(cols):
            print(final_map[x, y])
            if np.all(final_map[x, y] == [255, 255, 255]):
                #print('hi')
                return_map[x, y] = 0
            elif np.all(final_map[x, y] == [0, 0, 0]):
                #print("hi")
                return_map[x, y] = 1
            elif np.all(final_map[x, y] == [255, 75, 15]):
                return_map[x, y] = 2
                #count+=1
        
    # print(return_map)
    return return_map
    
def Starter():
    input_img = cv2.imread('img.png')
    # cv2.imshow('hi', input_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    objects = ObjectDetector(net, input_img)
    final_map = DisplayObjects(input_img, objects, 0.25)
    data = Encoder(final_map)
    print(data)
    return data

# Starter()
    
 
        
                