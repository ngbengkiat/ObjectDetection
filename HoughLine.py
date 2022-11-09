# importing OpenCV library
import cv2 as cv
import numpy as np
  


def draw_lines(img, houghLines, color=[0, 255, 0], thickness=2):
    for line in houghLines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
 
            cv.line(img,(x1,y1),(x2,y2),color,thickness)   
                
 
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv.addWeighted(initial_img, α, img, β, λ)

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
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (11, 11), 0)
    edges_image = cv.Canny(blurred_image, 80, 160)
    rho_resolution = 1
    theta_resolution = np.pi/180
    threshold = 80
 
    hough_lines = cv.HoughLines(edges_image, rho_resolution , theta_resolution , threshold)
 
    hough_lines_image = np.zeros_like(image)
    if hough_lines is not None:
        draw_lines(hough_lines_image, hough_lines)
    original_image_with_hough_lines = weighted_img(hough_lines_image,image)
    
    cv.imshow("original", image)
    cv.imshow("blurred", blurred_image)
    cv.imshow("edge", edges_image)
    cv.imshow("hough_edge", hough_lines_image)
  
    # saving image in local storage
    #cv.imwrite("GeeksForGeeks.jpg", image)
  
    # If keyboard interrupt occurs, destroy image 
    # window
    if cv.waitKey(10) == 27:
        cv.destroyAllWindows()
        break