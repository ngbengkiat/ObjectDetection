# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:08:23 2022

@author: nbk
"""

import cv2
import numpy as np

# HSV Control using Trackbars
# read a colourful image
video_src = 0
cam = cv2.VideoCapture(video_src)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #1280x720
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#img = cv2.imread('Images2\\testimage2.jpg')
#img = cv2.resize(img, (320,280))

# convert BGR image to HSV

# define a null callback function for Trackbar
def null(x):
    pass

# create six trackbars for R, G and B - lower and higher masking limits 
cv2.namedWindow('RGB')
# arguments: trackbar_name, window_name, default_value, max_value, callback_fn
cv2.createTrackbar("Rlo", "RGB", 0, 255, null)
cv2.createTrackbar("Rhi", "RGB", 255, 255, null)
cv2.createTrackbar("Glo", "RGB", 0, 255, null)
cv2.createTrackbar("Ghi", "RGB", 255, 255, null)
cv2.createTrackbar("Blo", "RGB", 0, 255, null)
cv2.createTrackbar("Bhi", "RGB", 255, 255, null)

while True:
    frame_got, img = cam.read()
    if frame_got is False:
        break
    
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[: ,: , 0] = cv2.equalizeHist(ycrcb_img[: ,: , 0])
    img_equ = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

   
    # read the Trackbar positions
    rl = cv2.getTrackbarPos('Rlo','RGB')
    rh = cv2.getTrackbarPos('Rhi','RGB')
    gl = cv2.getTrackbarPos('Glo','RGB')
    gh = cv2.getTrackbarPos('Ghi','RGB')
    bl = cv2.getTrackbarPos('Blo','RGB')
    bh = cv2.getTrackbarPos('Bhi','RGB')
    # create a manually controlled mask
    # arguments: hsv_image, lower_trackbars, higher_trackbars
    mask = cv2.inRange(hsv, np.array([bl, gl, rl]), np.array([bh, gh, rh]))
    # derive masked image using bitwise_and method
    final = cv2.bitwise_and(img, img, mask=mask)
    
    Hori = np.concatenate((img, final), axis=1)
    # display image, mask and masked_image 
    cv2.imshow('RGB', Hori)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows() 