import cv2
import numpy as np

cam_port = 0
cam = cv2.VideoCapture(cam_port)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 250;

# Filter by Area.
params.filterByArea = True
params.minArea = 500

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.3

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

params.blobColor = True
params.blobColor = 0

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)
while True:
    result, image = cam.read()
    if image is None :
        continue
    # Detect blobs.
    keypoints = detector.detect(image)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break

