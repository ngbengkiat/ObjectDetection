import cv2 as cv2
import matplotlib.pyplot as plt
#%matplotlib inline

#reading image
img1 = cv2.imread('Images2\Red.jpg')  
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#keypoints
sift = cv2.xfeatures2d.SIFT_create()


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    # Get webcam images
    ret, frame = cap.read()
    if ret is False:
        break
    keypoints_1, descriptors_1 = sift.detectAndCompute(frame,None)

    frame = cv2.drawKeypoints(gray1,keypoints_1,frame)
    plt.imshow(frame)
    cv2.imshow('Object Detector using ORB', frame)
    if cv2.waitKey(1) == 27: #13 is the ESC Key
        break
cap.release()
cv2.destroyAllWindows()