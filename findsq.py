# importing OpenCV library
import cv2 as cv
import numpy as np


# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that

#img = cv.imread('Images2\newblue.png')
#print(img)
#cv.imshow("Template", img)

cam_port = 0
cam = cv.VideoCapture(cam_port)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 800)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    result, image = cam.read()
    # showing result, it take frame name and image 
    # output        
    if result is False:
            break
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
    edges_image = cv.Canny(blurred_image, 80, 160)
    contours, _ = cv.findContours(edges_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    newimg = np.zeros_like(image)
    cnt_img = cv.drawContours(newimg, contours, -1, (0, 255, 0), 3)
    for cnt in contours:
        perimeter = int(cv.arcLength(cnt,True))
        approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
        if perimeter>500 and len(approx) > 6:
            cv.drawContours(image, [approx], -1, (0, 255, 0), 2)
    cv.imshow("original", image)
    cv.imshow("blurred", blurred_image)
    cv.imshow("edge", edges_image)
    cv.imshow("contour", cnt_img)
  
    # saving image in local storage
    #cv.imwrite("GeeksForGeeks.jpg", image)
  
    # If keyboard interrupt occurs, destroy image 
    # window
    if cv.waitKey(10) == 27:
        cv.destroyAllWindows()
        break