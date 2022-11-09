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
ap.add_argument("-t", "--template", type=str, required=True,
	help="path to template image")
ap.add_argument("-b", "--threshold", type=float, default=0.8,
	help="threshold for multi-template matching")
args = vars(ap.parse_args())

# load the input image and template image from disk, then grab the
# template image spatial dimensions
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
(tH, tW) = template.shape[:2]

image2 = cv2.blur(image,(5,5));
template2 = cv2.blur(template,(5,5));

# display the  image and template to our screen
cv2.imshow("Image", image2)
cv2.imshow("Template", template2)

# convert both the image and template to grayscale
imageGray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
# perform template matching
print("[INFO] performing template matching...")
result = cv2.matchTemplate(image2, template2,
	cv2.TM_CCOEFF_NORMED)

# find all locations in the result map where the matched value is
# greater than the threshold, then clone our original image so we
# can draw on it
(yCoords, xCoords) = np.where(result >= args["threshold"])
clone = image.copy()
print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))
# loop over our starting (x, y)-coordinates
for (x, y) in zip(xCoords, yCoords):
	# draw the bounding box on the image
	cv2.rectangle(clone, (x, y), (x + tW, y + tH),
		(255, 0, 0), 3)
# show our output image *before* applying non-maxima suppression
cv2.imshow("Before NMS", clone)
cv2.waitKey(0)

# initialize our list of rectangles
rects = []
# loop over the starting (x, y)-coordinates again
for (x, y) in zip(xCoords, yCoords):
	# update our list of rectangles
	rects.append((x, y, x + tW, y + tH))
# apply non-maxima suppression to the rectangles
pick = non_max_suppression(np.array(rects))
print("[INFO] {} matched locations *after* NMS".format(len(pick)))
# loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
	# draw the bounding box on the image
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(255, 0, 0), 3)
# show the output image
cv2.imshow("After NMS", image)
cv2.waitKey(0)