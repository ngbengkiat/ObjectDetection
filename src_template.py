# importing OpenCV library
import cv2 as cv
  
# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that
img = cv.imread('Images2\newblue.png')
print(img)
   
#cv.imshow("Template", img)

cam_port = 0
cam = cv.VideoCapture(cam_port)
#cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
#cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    result, image = cam.read()
    # showing result, it take frame name and image 
    # output        
    if result is False:
            break
    cv.imshow("GeeksForGeeks", image)
  
    # saving image in local storage
    cv.imwrite("GeeksForGeeks.jpg", image)
  
    # If keyboard interrupt occurs, destroy image 
    # window
    if cv.waitKey(10) == 27:
        cv.destroyAllWindows()
        break
  
