# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import cv2
#while  True:
command="fswebcam -d /dev/video0 -r 600*600 /home/wyx/桌面/one_re/photo1.jpg"
os.system(command)
time.sleep(2)
face_cascade=cv2.CascadeClassifier("/opt/ros/kinetic/share/OpenCV-3.3.1-dev/haarcascades/haarcascade_frontalface_alt2.xml")
img=cv2.imread("/home/wyx/桌面/one_re/photo1.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray)
if(len(faces) > 0):
	face_re = True	
else:
	face_re = False
time.sleep(2)

