# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:14:24 2022

@author: nbk
"""

# import the necessary pages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image where we'll apply template matching")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to output image")

args = vars(ap.parse_args())

# load the input image and template image from disk, then grab the
# template image spatial dimensions
print("[INFO] loading images...")
image = cv2.imread(args["image"])

image2 = cv2.blur(image,(9,9));

# display the  image and template to our screen
cv2.imshow("Image", image2)

cv2.imwrite(args["output"], image2);
cv2.waitKey(0)