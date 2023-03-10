# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:59:23 2023

@author: Asus
"""
import os
import urllib
import tarfile

modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "coco_class_labels.txt"
if not os.path.isdir('models'):
    os.mkdir("models")
    if not os.path.isfile(modelFile):
        os.chdir("models")
        # Download the tensorflow Model
        urllib.request.urlretrieve('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz', 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
        
        # Uncompress the file
        #!tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
        tarfile.open(name = 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz', mode = 'r')
            
        # Delete the tar.gz file
        os.remove('ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
            
        #Come back to the previous directory
        os.chdir("..")