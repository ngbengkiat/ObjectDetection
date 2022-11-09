import cv2
import numpy as np


def ORB_detector(new_image, template):
    # Function that compares input image to template
    # It then returns the number of ORB matches between them
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image1', image1)
    cv2.waitKey(1)
    
    # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB_create(1000, 1.2)

    # Detect keypoints of original image
    (kp1, des1) = orb.detectAndCompute(image1, None)
    img2 = cv2.drawKeypoints(image1, kp1, None, color=(0,255,0), flags=0)
    
    cv2.imshow('image2', img2)
    cv2.waitKey(1)
    
    # Detect keypoints of rotated image
    (kp2, des2) = orb.detectAndCompute(template, None)
    img3 = cv2.drawKeypoints(template, kp2, None, color=(0,255,0), flags=0)
    
    cv2.imshow('image3', img3)
    cv2.waitKey(1)
    
    if des2 is None:
        print('None')
        return 0
    
    # Create matcher 
    # Note we're no longer using Flannbased matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Do matching
    matches = bf.match(des1,des2)

    # Sort the matches based on distance.  Least distance
    # is better
    matches = sorted(matches, key=lambda val: val.distance)
    return len(matches)


print("[INFO] loading images...")
# Load our image template, this is our reference image
image_template = cv2.imread("Images2\\newgurney.png", cv2.IMREAD_GRAYSCALE) 
cv2.imshow('Template', image_template)
cv2.waitKey(1)
# image_template = cv2.imread('images/kitkat.jpg', 0) 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Get webcam images
    ret, frame = cap.read()
    if ret is False:
        break
    # Get height and width of webcam frame
    height, width = frame.shape[:2]

    # Define ROI Box Dimensions (Note some of these things should be outside the loop)
    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))

    # Draw rectangular window for our region of interest
    cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 3)

    # Crop window of observation we defined above
    cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]

    # Flip frame orientation horizontally
    frame = cv2.flip(frame,1)
    cv2.imshow('MMM', cropped)
    cv2.waitKey(1)

    # Get number of ORB matches 
    matches = ORB_detector(frame, image_template)

    # Display status string showing the current no. of matches 
    output_string = "Matches = " + str(matches)
    cv2.putText(frame, output_string, (50,450), cv2.FONT_HERSHEY_COMPLEX, 2, (250,0,150), 2)

    # Our threshold to indicate object deteciton
    # For new images or lightening conditions you may need to experiment a bit 
    # Note: The ORB detector to get the top 1000 matches, 350 is essentially a min 35% match
    threshold = 10

    # If matches exceed our threshold then object has been detected
    if matches > threshold:
        cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
        cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)

    cv2.imshow('Object Detector using ORB', frame)
    if cv2.waitKey(1) == 27: #13 is the ESC Key
        break

cap.release()
cv2.destroyAllWindows()