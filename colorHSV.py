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
[166,5,132,255,70,255,],
[23,43,109,255,112,255,],
[88,113,109,255,128,255,],
[40,50,60,70,80,90,],
]
)
    
trackNames = "H_lo", "H_hi", "S_lo", "S_hi", "V_lo", "V_hi"
hsvRange = [180, 180, 255, 255, 255, 255]    

# create six trackbars for H, S and V - lower and higher masking limits 
winName = "HSV"
imgName = "HSV"
cv2.namedWindow(winName)
cv2.resizeWindow(winName, 300, 500)
font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.7
fontColor = (0,0,255)

for j in range(6):
    s = trackNames[j]
    cv2.createTrackbar(s, winName, hsvVal[color][j], hsvRange[j], null)
cv2.createTrackbar("Color", winName, 0, numOfColors-1, null)
    
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
    if hsvVal[color][1] >= hsvVal[color][0]: 
        mask = cv2.inRange(img_hsv, np.array([hsvVal[color][0], hsvVal[color][2], hsvVal[color][4]]), 
                           np.array([hsvVal[color][1], hsvVal[color][3], hsvVal[color][5]]))
    else:
        mask0 = cv2.inRange(img_hsv, np.array([hsvVal[color][0], hsvVal[color][2], hsvVal[color][4]]), 
                           np.array([hsvRange[0], hsvVal[color][3], hsvVal[color][5]]))
        mask1 = cv2.inRange(img_hsv, np.array([0, hsvVal[color][2], hsvVal[color][4]]), 
                           np.array([hsvVal[color][1], hsvVal[color][3], hsvVal[color][5]]))
        mask = mask0 | mask1

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = int(cv2.contourArea(cnt))

        if area < 500:
            continue
        perimeter = int(cv2.arcLength(cnt,True))

        moment = cv2.moments(cnt)
        if moment['m00'] != 0.0:
            mx = int(moment['m10']/moment['m00'])
            my = int(moment['m01']/moment['m00'])
            # Ellipse
            e = cv2.fitEllipse(cnt)
            #cv2.ellipse(tmpArea, e, (0, 255, 0), 2)

            # Principal axis
            xp1 = int(np.round(mx + e[1][1] / 2 * np.cos((e[2] + 90) * np.pi / 180.0)))
            yp1 = int(np.round(my + e[1][1] / 2 * np.sin((e[2] + 90) * np.pi / 180.0)))
            xp2 = int(np.round(mx + e[1][1] / 2 * np.cos((e[2] - 90) * np.pi / 180.0)))
            yp2 = int(np.round(my + e[1][1] / 2 * np.sin((e[2] - 90) * np.pi / 180.0)))


        x1,y1,w,h = cv2.boundingRect(cnt)
        x2 = x1+w
        y2 = y1+h
        xc = (x1+x2)/2
        yc = (y1+y2)/2
        if area > 500:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            cv2.drawContours(mask_bgr, [approx], -1, (0, 255, 0), 2)
            #cv2.drawContours(mask_bgr, cnt, -1, (0, 255, 0), 2)
            cv2.line(mask_bgr, (xp1, yp1), (xp2, yp2), (255, 255, 0), 2)
            
            cv2.putText(mask_bgr, "area="+str(area), (mx,my), font, fontScale, fontColor, 2)
            cv2.putText(mask_bgr, "peri="+str(perimeter), (mx,my+20), font, fontScale, fontColor, 2)
            cv2.putText(mask_bgr, "angl="+str(e[2]), (mx,my+40), font, fontScale, fontColor, 2)
            cv2.putText(mask_bgr, "erro="+str(mx-cam.get(cv2.CAP_PROP_FRAME_WIDTH)/2), (mx,my+60), font, fontScale, fontColor, 2)
    
    # derive masked image using bitwise_and method
    #final = cv2.bitwise_and(img, img, mask=mask)
    
    Hori = np.concatenate((img, mask_bgr), axis=1)

    cv2.imshow(imgName, Hori)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cam.release()    
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