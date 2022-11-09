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
#ap.add_argument("-i", "--image", type=str, required=True,
#	help="path to input image where we'll apply template matching")
ap.add_argument("-t", "--template", type=str, required=True,
	help="path to template image")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to mask image")
args = vars(ap.parse_args())

# load the input image and template image from disk, then grab the
# template image spatial dimensions
print("[INFO] loading images...")
#image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
(tH, tW) = template.shape[:2]
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
print(template)

print("gray")
print(templateGray[0])
print(templateGray[1])
print(templateGray[2])
print(templateGray[3])
cv2.imshow("Template", template)
cv2.imshow("Gray", templateGray)


cv2.waitKey(1000) 
    
cv2.destroyAllWindows()
