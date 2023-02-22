import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

while True:
    # Get webcam images
    ret, image = cap.read()
    if ret is False:
        break

    gray0= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray0, (5, 5), 0)
    edges_image = cv2.Canny(blurred_image, 200, 320)

    gray= np.float32(edges_image)

    harris_corners= cv2.cornerHarris(gray, 2, 3, 0.1)

    kernel= np.ones((3,3  ), np.uint8)

    harris_corners= cv2.dilate(harris_corners, kernel, iterations= 2)

    image[harris_corners > 0.025 * harris_corners.max()]= [255,127,127]

    cv2.imshow('Edge', edges_image)
    cv2.imshow('Harris Corners', image)
    if cv2.waitKey(1) == 27: #13 is the ESC Key
        break

cap.release()
cv2.destroyAllWindows()