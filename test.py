import cv2
import numpy as np

# HSV Control using Trackbars
# read a colourful image
video_src = 0
cam = cv2.VideoCapture(video_src)
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #1280x720
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#img = cv2.imread('Images2\\testimage2.jpg')
#img = cv2.resize(img, (320,280))

# convert BGR image to HSV

# define a null callback function for Trackbar
def null(x):
    pass

numOfColors = 4
color = 0
    
hsvVal = np.array(
[
[0,180,0,255,0,255,],
[20,30,40,50,60,70,],
[30,40,50,60,70,80,],
[40,50,60,70,80,90,],
]
)
    
trackNames = "H_lo", "H_hi", "S_lo", "S_hi", "V_lo", "V_hi"
hsvRange = [180, 180, 255, 255, 255, 255]    

# create six trackbars for H, S and V - lower and higher masking limits 
winName = "HSV"
imgName = "Image"
cv2.namedWindow(winName)
cv2.resizeWindow(winName, 300, 500)


cv2.createTrackbar("Color", winName, 0, numOfColors-1, null)

for j in range(6):
    s = trackNames[j]
    cv2.createTrackbar(s, winName, hsvVal[color][j], hsvRange[j], null)
    
while True:
    frame_got, img = cam.read()
    if frame_got is False:
        break
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   
    # read the Trackbar positions
    #First tracker select the color to set
    newColor = cv2.getTrackbarPos("Color",winName)
    if newColor != color:
        color = newColor
        for j in range(6):
            s = trackNames[j]
            cv2.setTrackbarPos(s,winName, hsvVal[color][j])
            
    #Next set of sliders for HSV setting            
    for j in range(6):
        s = trackNames[j]
        hsvVal[color][j] = cv2.getTrackbarPos(s,winName)


    # create a manually controlled mask
    # arguments: hsv_image, lower_trackbars, higher_trackbars
    mask = cv2.inRange(img_hsv, np.array([hsvVal[color][0], hsvVal[color][2], hsvVal[color][4]]), 
                       np.array([hsvVal[color][1], hsvVal[color][3], hsvVal[color][5]]))

    # derive masked image using bitwise_and method
    final = cv2.bitwise_and(img, img, mask=mask)
    
    Hori = np.concatenate((img, final), axis=1)

    #cv2.imshow('Equalised', img_equ)
    cv2.imshow(imgName, Hori)
    #cv2.imshow('Masked Image', final)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows() 

#printout for easy update
print("hsvVal = np.array(")
print("[")
for i in range(numOfColors):
    print("[", end = '')
    for j in range(6):
        print(hsvVal[i][j],  end='')
        print(',', end='')
    print('],')
print('],')    