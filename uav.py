#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:42:59 2023

@author: chushiyun
"""

from ultralytics import YOLO
import cv2
import time

# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   #data='uav.yaml',
   data='/root/code/uav/uav.yaml',
   imgsz=640,
   epochs=10,
   batch=5,
   name='yolov8n_v8_50e'
)

