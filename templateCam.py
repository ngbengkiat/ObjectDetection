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

#                Filename            BGR           text
templateInfoList = [ ["Images2\Red.jpg",  (0, 0, 255),  "Red"],
                 ["Images2\Blue.jpg", (255, 0, 0), "Blue"],
                 ["Images2\Yellow.jpg", (0, 255, 255), "Yellow"],
                 ["Images2\Gurney.jpg", (0, 0, 0), "Gurney"] ]

blurSize = (5,5)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", type=str, required=True,
#	help="path to input image where we'll apply template matching")
#ap.add_argument("-t", "--template", type=str, required=True,
#	help="path to template image")

ap.add_argument("-b", "--threshold", type=float, default=0.8,
	help="threshold for multi-template matching")
args = vars(ap.parse_args())

# load the input image and template image from disk, then grab the
# template image spatial dimensions
print("[INFO] loading images...")

template = []
templateBlur = []
templateSize = []
templateColor = []
numOfTemplate = len(templateInfoList)
for i in range(numOfTemplate):
    template.append(cv2.imread(templateInfoList[i][0]))
    templateSize.append(template[i].shape[:2])
    templateBlur.append(cv2.blur(template[i], blurSize) );
    templateColor.append(templateInfoList[i][1])
    
cv2.imshow("Template", templateBlur[0])

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
while True:
    # display the  image to our screen
    frame_got, image = cam.read()
    if frame_got is False:
        break
    imageBlur = cv2.blur(image, blurSize);
    cv2.imshow("Image", image)
    clone = image.copy()
    
    for i in range(numOfTemplate):
        (tW, tH) = templateSize[i]
        # perform template matching
        print("[INFO] performing template matching...")
        #result = cv2.matchTemplate(image, template[i],
        #                           cv2.TM_CCOEFF_NORMED)
        result = cv2.matchTemplate(imageBlur, templateBlur[i],
                                   cv2.TM_CCOEFF_NORMED)

        # find all locations in the result map where the matched value is
        # greater than the threshold, then clone our original image so we
        # can draw on it
        (yCoords, xCoords) = np.where(result >= args["threshold"])


        print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))
        # loop over our starting (x, y)-coordinates
        for (x, y) in zip(xCoords, yCoords):
            # draw the bounding box on the image
            cv2.rectangle(clone, (x, y), (x + tW, y + tH), templateColor[i], 3)
    

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
	        cv2.rectangle(imageBlur, (startX, startY), (endX, endY),
		        templateColor[i], 3)

    # show our output image *before* applying non-maxima suppression
    cv2.imshow("Before NMS", clone)
    # show the output image
    cv2.imshow("After NMS", imageBlur)
    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break