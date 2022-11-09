import cv2 as cv
import numpy as np
#import math
def null(x):
    pass

cam_port = 1
cam = cv.VideoCapture(cam_port)
#cam.set(cv.CAP_PROP_FRAME_WIDTH, 800)
#cam.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
cv.namedWindow('HOUGH')
cv.createTrackbar("CannyL", "HOUGH", 110, 1000, null)
cv.createTrackbar("CannyH", "HOUGH", 210, 1000, null)
cv.createTrackbar("HoughT", "HOUGH", 30, 255, null)
cv.createTrackbar("HoughMin", "HOUGH", 40, 255, null)
cv.createTrackbar("HoughGap", "HOUGH", 30, 255, null)

while True:
    result, image = cam.read()
    if image is None :
        continue
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    #img_dilation = cv.dilate(gray_image, kernel, iterations=3) 
    #blurred_image = cv.bilateralFilter(gray_image, 7, 50, 50)
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)    
    canny_l = cv.getTrackbarPos('CannyL','HOUGH')
    canny_h = cv.getTrackbarPos('CannyH','HOUGH')
    hough_T = cv.getTrackbarPos('HoughT','HOUGH')
    hough_Min = cv.getTrackbarPos('HoughMin','HOUGH')
    hough_Gap = cv.getTrackbarPos('HoughGap','HOUGH')
    canny_img = cv.Canny(blurred_image, canny_l, canny_h, apertureSize=3, L2gradient=True)


    linesP = cv.HoughLinesP(canny_img, rho=1, theta=np.pi/180, threshold=hough_T, minLineLength=hough_Min, maxLineGap=hough_Gap)
    out_img = cv.cvtColor(canny_img, cv.COLOR_GRAY2BGR)

    if linesP is not None:
        for i in range(0, len(linesP)):
            lin = linesP[i][0]
            cv.line(image, (lin[0], lin[1]), (lin[2], lin[3]), (0,0,255), 2, cv.LINE_AA)
            cv.line(out_img, (lin[0], lin[1]), (lin[2], lin[3]), (0,0,255), 2, cv.LINE_AA)

    cv.imshow("image", image)
    cv.imshow("edge", out_img)
  
    # saving image in local storage
    #cv.imwrite("GeeksForGeeks.jpg", image)
  
    # If keyboard interrupt occurs, destroy image 
    # window
    if cv.waitKey(10) == 27:
        cv.destroyAllWindows()
        break