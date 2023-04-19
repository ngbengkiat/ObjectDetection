import os
import argparse
import cv2
import numpy as np
import math
import sys
import time
from threading import Thread
import threading
import importlib.util
from networktables import NetworkTables
import imutils
from math import atan2, sin, cos, pi, sqrt
def connectionListener(connected,info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()
cond = threading.Condition()
notified = [False]

NetworkTables.initialize(server='10.42.1.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
 print("Waiting")
 if not notified[0]:
     cond.wait()
print("Connected!")

sd = NetworkTables.getTable("Shuffleboard/Vision")
pointsTable = NetworkTables.getTable("Shuffleboard/Points")

global redDistance, greenDistance, blueDistance, trolley, objects, array, line

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(800,600),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(cv2.CAP_PROP_FPS, framerate)
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=False,default='JagabeeSis')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='model.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labels.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.65)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='800x600')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation


# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
hsvVal = np.loadtxt('hsvVal.txt',delimiter=',')


# Array Variables
def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)

  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
  ## [visualization1]

def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]

  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]

  ## [visualization]
  # Draw the principal components
  cv2.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 5)

  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]

  # Label with the rotation angle
  label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
#   textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
#   cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

  return angle
def get_box_xy(box, imH, imW):
    ymin = int(max(1, (box[0] * imH)))
    xmin = int(max(1, (box[1] * imW)))
    ymax = int(min(imH, (box[2] * imH)))
    xmax = int(min(imW, (box[3] * imW)))
    return ymin, xmin, ymax, xmax

def modelResult(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
    return boxes, classes, scores


def sendValues(mode):


    if mode == 0:
        sd.putNumberArray("line" , line)
    elif mode == 1:
        sd.putNumberArray("objects", objects)
    elif mode == 2:
        sd.putNumberArray('WOB', array)

    elif mode == 3:
        sd.putNumberArray("RedTarget", redDistance)
        sd.putNumberArray("GreenTarget", greenDistance)
        sd.putNumberArray("BlueTarget", blueDistance)
    elif mode == 4:
        pointsTable.putNumberArray("Trolley", trolley)
        pointsTable.putNumberArray("RedTarget", redDistance)
        pointsTable.putNumberArray("GreenTarget", greenDistance)
        pointsTable.putNumberArray("BlueTarget", blueDistance)
        pointsTable.putNumberArray("Bin", yellowBin)
    elif mode == 5:
        sd.putNumberArray("line" , line)
#     sd.putNumberArray("Trolley", trolley)
    if mode != -1:
        NetworkTables.flush()
def createValues():
    global redDistance, greenDistance, blueDistance, trolley, objects, array, line, yellowBin
    redDistance = np.zeros(2, dtype=np.float64)
    greenDistance = np.zeros(2, dtype=np.float64)
    blueDistance = np.zeros(2, dtype=np.float64)
    trolley = np.zeros(6, dtype=np.float64)
    objects = np.zeros(12, dtype=int)
    array = np.zeros(9, dtype=int)
    line = np.zeros(3, dtype=float)
    yellowBin = np.zeros(6,dtype=float)
def resetValues():

    redDistance = np.zeros(2, dtype=np.float64)
    greenDistance = np.zeros(2, dtype=np.float64)
    blueDistance = np.zeros(2, dtype=np.float64)
    trolley = np.zeros(2, dtype=np.float64)
    objects = np.zeros(12, dtype=int)
    array = np.zeros(9, dtype=int)
    line = np.zeros(3, dtype=float)
    yellowBin = np.zeros(6,dtype=float)
def main():
    global redDistance, greenDistance, blueDistance, trolley, objects, array, line, yellowBin,frame_rate_calc

    red_cX, red_cY,green_cX, green_cY, blue_cX, blue_cY = 0,0,0,0,0,0
    area_threshold = 5000
    # Output dimensions for perspective correction
    output_height = 640
    output_width= 440 # output_height * 89/130

    # Perspective Transformation points
    src_points = np.float32([[160,20],[450,20],[0,480],[640,480]])
    src_tf_points = np.float32([[215,40],[600,40],[0,600],[800,600]])
    dst_points = np.float32([[0, 0], [output_width, 0], [0, output_height], [output_width, output_height]])
    # Perspective Transformation convert pixel to meters
    X_to_m = 1.00/output_width
    Y_to_m = 1.40/output_height

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    cX, cY, cW = 0 , 0, 0

    bitAND = np.zeros((6,7), dtype=np.uint8)
    trolleyCount = 0
    binCount = 0
    trolleyRatio = 0.9
    createValues()
    while True:

        cvMode = sd.getNumber("cvMode",-1)
#         cvMode = 0
        frame1 = videostream.read()
        if cvMode == -1:
            cv2.imshow('Live Feed', frame1)      #live camera code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif cvMode == 0:
            debug=0
            img = videostream.read()
            img = imutils.resize(img, width = 400)
            t1 = cv2.getTickCount()
            # Retrieve from shuffleboard which color to detect
            # Black, Red, Green Blue, 0 , 1, 2, 3 Respectively
            color = int(sd.getNumber('ColorMode',0))
#             color = 1
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             mask = cv2.inRange(img_hsv, np.array([hsvVal[color][0], hsvVal[color][2], hsvVal[color][4]]),
#                                np.array([hsvVal[color][1], hsvVal[color][3], hsvVal[color][5]]))

            if hsvVal[color][1] >= hsvVal[color][0]:
                mask = cv2.inRange(img_hsv, np.array([hsvVal[color][0], hsvVal[color][2], hsvVal[color][4]]),
                                   np.array([hsvVal[color][1], hsvVal[color][3], hsvVal[color][5]]))
            else:
                mask0 = cv2.inRange(img_hsv, np.array([hsvVal[color][0], hsvVal[color][2], hsvVal[color][4]]),
                                   np.array([180, hsvVal[color][3], hsvVal[color][5]]))
                mask1 = cv2.inRange(img_hsv, np.array([0, hsvVal[color][2], hsvVal[color][4]]),
                                   np.array([hsvVal[color][1], hsvVal[color][3], hsvVal[color][5]]))
                mask = mask0 | mask1

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if debug==1:
                result = img


            try:
                _, contours,_  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                img_center = np.array([img.shape[1] / 2, img.shape[0]])

                # Find the nearest contour to the center
                closest_contour = None
                nearest_distance = float("inf")
                largest_area = float("-inf")
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                for contour in contours:
                    moments = cv2.moments(contour)
                    if moments["m00"] == 0:
                        continue
                    contour_center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
                    distance = np.linalg.norm(contour_center - img_center)
                    area = cv2.contourArea(contour)
                    if distance < nearest_distance and area > largest_area and area > 1000:
                        closest_contour = contour
                        largest_area = area
                        nearest_distance = distance
                print(largest_area)
                extLeft = tuple(closest_contour[closest_contour[:, :, 0].argmin()][0])
                extRight = tuple(closest_contour[closest_contour[:, :, 0].argmax()][0])
                M = cv2.moments(closest_contour)
                cX = line[0] = int(M["m10"] / M["m00"])
                cY = line[1] = int(M["m01"] / M["m00"])

                blank = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

                ht = 10;
                if(extRight[1] < extLeft[1]): #if right is higher

                    cv2.rectangle(mask_bgr, (extRight[0]+ht, extRight[1]-ht),(extLeft[0]-ht, extLeft[1]+ht), (255,0,0),-1)
                    cv2.rectangle(blank, (extRight[0]+ht, extRight[1]-ht),(extLeft[0]-ht, extLeft[1]+ht), (255,0,0),-1)
                else:
                    cv2.rectangle(mask_bgr, (extLeft[0]-ht, extLeft[1]-ht), (extRight[0]+ht, extRight[1]+ht), (255,0,0),-1)
                    cv2.rectangle(blank, (extLeft[0]-ht, extLeft[1]-ht), (extRight[0]+ht, extRight[1]+ht), (255,0,0),-1)

                bitAND = cv2.bitwise_and(blank, mask)
                _, cnts,_  = cv2.findContours(bitAND, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
                cW = line[2] = getOrientation(c, mask_bgr)

                if debug==1:
                    cv2.circle(result, (cX,cY), 7, (255,255,255), -1)
                    cv2.putText(result, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
                    cv2.putText(result,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                    cv2.imshow("result", result)

                cv2.circle(mask_bgr, extLeft, 4, (0,0,255), -1)
                cv2.circle(mask_bgr, extRight, 4, (0,0,255), -1)
            except Exception as e:
                cX, cY, cW = line[0], line[1], line[2] = 0,0,0
                print(e)




            #cv2.imshow("AND", bitAND)
            cv2.imshow("mask_bgr", mask_bgr)
            #cv2.imshow("mask", mask)


            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            print("center X: {0} center Y: {1} W: {2}".format(cX,cY, cW))
            frame_rate = 30
            wait_time = int(1000/frame_rate)
            sendValues(cvMode)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        elif cvMode == 1:
            global imH, imW
            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            boxes, classes, scores = modelResult(frame)
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            objects[0], objects[3], objects[6], objects[9] = 0, 0, 0 , 0
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    cX = int((xmax+xmin)/2)
                    cY = int((ymax+ymin)/2)



                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.circle(frame, (cX,cY), 7, (255,255,255), -1)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index

                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'

                    if object_name == "CokeU":
                        objects[0] += 1
                        objects[1] = cX
                        objects[2] = cY

                    elif object_name == "Coke":
                        objects[3] += 1
                        objects[4] = cX
                        objects[5] = cY
                    elif object_name == "Dettol":
                        objects[6] += 1
                        objects[7] = cX
                        objects[8] = cY
                    elif object_name == "Jagabee":
                        objects[9] += 1
                        objects[10] = cX
                        objects[11] = cY


                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text





#             NetworkTables.flush()
            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1
            sd.putNumber("FPS",frame_rate_calc)
            sendValues(cvMode)
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
        elif cvMode == 2:
            t1 = cv2.getTickCount()
            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            boxes, classes, scores = modelResult(frame)

            cx_rgb = np.zeros((3,), dtype=int)
            dx = np.zeros((3,), dtype=int)
            rgb_obj = np.zeros((3, 3), dtype=int)
            imH, imW, _ = frame.shape
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    ymin, xmin, ymax, xmax = get_box_xy(boxes[i], imH, imW)

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 255), 2)

                    object_name = labels[int(classes[i])]
                    label = '%s: %d%%' % (object_name, int(
                        scores[i]*100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)  # Get font size
                    # Make sure not to draw label too close to top of window
                    label_ymin = max(ymin, labelSize[1] + 10)
                    # Draw white box to put label text in
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin +
                                labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  # Draw label text
                    if (classes[i] == 4):  # red
                        ymin, xmin, ymax, xmax = get_box_xy(boxes[i], imH, imW)
                        cx_rgb[0] = int((xmin+xmax)/2)

                    if (classes[i] == 5):  # red
                        ymin, xmin, ymax, xmax = get_box_xy(boxes[i], imH, imW)
                        cx_rgb[1] = int((xmin+xmax)/2)

                    if (classes[i] == 6):  # red
                        ymin, xmin, ymax, xmax = get_box_xy(boxes[i], imH, imW)
                        cx_rgb[2] = int((xmin+xmax)/2)



            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0) and object_name != "WOBq"):
                    obj_idx = classes[i] - 1

                    if (obj_idx < 2):
                        ymin, xmin, ymax, xmax = get_box_xy(boxes[i], imH, imW)
                        cx = (xmin+xmax)/2
                        # Cal x-dist from R G B
                        dx[0] = math.fabs(cx-cx_rgb[0])
                        dx[1] = math.fabs(cx-cx_rgb[1])
                        dx[2] = math.fabs(cx-cx_rgb[2])

                    # Find closest R G or B
                        rgb_idx = 0
                        if (dx[1] < dx[int(rgb_idx)]):
                            rgb_idx = 1
                        if (dx[2] < dx[int(rgb_idx)]):
                            rgb_idx = 2
                        rgb_obj[int(rgb_idx)][int(obj_idx)+1] += 1

                    # Draw color to show
                        if (rgb_idx == 0):  # red
                            cv2.rectangle(frame, (xmin, ymin),
                                            (xmax, ymax), (0, 0, 255), 2)
                        elif (rgb_idx == 1):  # green
                            cv2.rectangle(frame, (xmin, ymin),
                                            (xmax, ymax), (0, 255, 0), 2)
                        elif (rgb_idx == 2):  # blue
                            cv2.rectangle(frame, (xmin, ymin),
                                            (xmax, ymax), (255, 50, 50), 2)


            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            for i in range(3):
                size = len(rgb_obj[i])

                # Swapping
                temp = rgb_obj[i][0]
                rgb_obj[i][0] = rgb_obj[i][size - 1]
                rgb_obj[i][size - 1] = temp

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1
            print('Coke Dett Jaga') #used to be JDC
            print('R ', rgb_obj[0])
            print('G ', rgb_obj[1])
            print('B ', rgb_obj[2])

            newArray = np.reshape(rgb_obj, (1, 9))
            array = newArray.flatten('C')
            print(array)
            sendValues(cvMode)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        elif cvMode == 3:

            frame = imutils.resize(frame1, width = 640)
            print(frame.shape)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            # Apply the perspective transformation to the frame
            transformed_frame = cv2.warpPerspective(frame, M, (output_width, output_height))

            # Convert the frame from BGR to HSV
            hsv = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

            # Threshold the image to only select colors in the specified range
            red_mask = cv2.inRange(hsv, np.array([hsvVal[1][0], hsvVal[1][2], hsvVal[1][4]]),
                               np.array([hsvVal[1][1], hsvVal[1][3], hsvVal[1][5]]))
            green_mask = cv2.inRange(hsv, np.array([hsvVal[2][0], hsvVal[2][2], hsvVal[2][4]]),
                               np.array([hsvVal[2][1], hsvVal[2][3], hsvVal[2][5]]))
            blue_mask = cv2.inRange(hsv, np.array([hsvVal[3][0], hsvVal[3][2], hsvVal[3][4]]),
                               np.array([hsvVal[3][1], hsvVal[3][3], hsvVal[3][5]]))

            result = transformed_frame.copy()
            try:

                    red_contours = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    red_contours = red_contours[0] if len(red_contours) == 2 else red_contours[1]
                    red_c = max(red_contours, key = cv2.contourArea)
                    red_area = int(cv2.contourArea(red_c))

                    if red_area > area_threshold:
                        red_moments = cv2.moments(red_c)

                        red_cX = int(red_moments["m10"] / red_moments["m00"])
                        red_cY = int(red_moments["m01"] / red_moments["m00"])

                        cv2.circle(result, (red_cX,red_cY), 7, (255,255,255), -1)
                    else:
                        red_cX, red_cY = 0, 0
            except Exception as e:
                    red_cX, red_cY = 0, 0
                    redDistance[0] = redDistance[1] = 0

                    print("No red target")
            try:
                    green_contours = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    green_contours = green_contours[0] if len(green_contours) == 2 else green_contours[1]
                    green_c = max(green_contours, key = cv2.contourArea)
                    green_area = int(cv2.contourArea(green_c))
                    if green_area > area_threshold:
                        green_moments = cv2.moments(green_c)

                        green_cX = int(green_moments["m10"] / green_moments["m00"])
                        green_cY = int(green_moments["m01"] / green_moments["m00"])

                        cv2.circle(result, (green_cX,green_cY), 7, (255,255,255), -1)
                    else:
                        green_cX, green_cY = 0, 0


            except Exception as e:
                    green_cX, green_cY = 0, 0
                    greenDistance[0] = greenDistance[1] = 0

                    print("No green detected")
            try:
                    blue_contours = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    blue_contours = blue_contours[0] if len(blue_contours) == 2 else blue_contours[1]

                    blue_c = max(blue_contours, key = cv2.contourArea)
                    blue_area = int(cv2.contourArea(blue_c))
                    if blue_area > area_threshold:
                        blue_moments = cv2.moments(blue_c)

                        blue_cX = int(blue_moments["m10"] / blue_moments["m00"])
                        blue_cY = int(blue_moments["m01"] / blue_moments["m00"])

                        cv2.circle(result, (blue_cX,blue_cY), 7, (255,255,255), -1)
                    else:
                        blue_cX, blue_cY = 0, 0
            except Exception as e:
                    blue_cX, blue_cY = 0, 0
                    blueDistance[0] = blueDistance[1] = 0

                    print("No Blue Detected")


            if red_cX and red_cY:
                redDistance[0] = (red_cX - output_width/2 ) * X_to_m
                redDistance[1] = (output_width - red_cY ) * Y_to_m

            else:
                redDistance[0] = redDistance[1] = 0

            if green_cX and green_cY:
                greenDistance[0] = (green_cX - output_width/2 ) * X_to_m
                greenDistance[1] = (output_width - green_cY ) * Y_to_m

            else:
                greenDistance[0] = greenDistance[1] = 0

            if blue_cX and blue_cY:
                blueDistance[0] = (blue_cX - output_width/2 ) * X_to_m
                blueDistance[1] = (output_width - blue_cY ) * Y_to_m

            else:
                blueDistance[0] = blueDistance[1] = 0


            print("Red Target Distance from camera: X: {0}, Y: {1}".format(redDistance[0],redDistance[1]))
            print("Green Target Distance from camera: X: {0}, Y: {1}".format(greenDistance[0],greenDistance[1]))
            print("Blue Target Distance from camera: X: {0}, Y: {1}".format(blueDistance[0],blueDistance[1]))

            cv2.imshow("Result", result)
            sendValues(cvMode)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif cvMode == 4:

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            M_tfModel = cv2.getPerspectiveTransform(src_tf_points, dst_points)
            boxes, classes, scores = modelResult(frame1)
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            img = imutils.resize(frame1, width = 800)
            color = 0
            transformed_frame = cv2.warpPerspective(frame1, M_tfModel, (output_width, output_height))
            img_hsv = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img_hsv, np.array([hsvVal[color][0], hsvVal[color][2], hsvVal[color][4]]),
                               np.array([hsvVal[color][1], hsvVal[color][3], hsvVal[color][5]]))

            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    cX = int((xmax+xmin)/2)
                    cY = int((ymax+ymin)/2)

                    cv2.rectangle(frame1, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.circle(frame1, (cX,cY), 7, (255,255,255), -1)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index

                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    point = np.array([[[cX,cY]]], dtype=np.float64)
                    transfromed_point = cv2.perspectiveTransform(point, M_tfModel).flatten('C')

                    if object_name == "Trolley":
                     trolley[trolleyCount + 0] = (transfromed_point[0] - output_width/2 ) * X_to_m
                     trolley[trolleyCount + 1] = (output_height - transfromed_point[1]  ) * Y_to_m * trolleyRatio
                     trolleyCount = trolleyCount + 2
                    elif object_name == "RedT":
                     redDistance[0] = (transfromed_point[0] - output_width/2 ) * X_to_m
                     redDistance[1] = (output_height  - transfromed_point[1]  ) * Y_to_m
                    elif object_name == "GreenT":
                     greenDistance[0] = (transfromed_point[0] - output_width/2 ) * X_to_m
                     greenDistance[1] = (output_height  - transfromed_point[1]  ) * Y_to_m
                    elif object_name == "BlueT":
                     blueDistance[0] = (transfromed_point[0] - output_width/2 ) * X_to_m
                     blueDistance[1] = (output_height  - transfromed_point[1]  ) * Y_to_m
                    elif object_name == "Bin":
                     yellowBin[binCount * 3 + 0] = (transfromed_point[0] - output_width/2 ) * X_to_m
                     yellowBin[binCount * 3 + 1] = (output_height  - transfromed_point[1]  ) * Y_to_m
                     yellowBin[binCount * 3 + 2] = 0
                     binCount += 1
#                      try:
#                         yellow_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                         yellow_contours = yellow_contours[0] if len(yellow_contours) == 2 else yellow_contours[1]
#                         yellow_c = max(yellow_contours, key = cv2.contourArea)
# #                         extLeft = tuple(yellow_c[yellow_c[:, :, 0].argmin()][0])
# #                         extRight = tuple(yellow_c[yellow_c[:, :, 0].argmax()][0])
# #                         yellowBin[binCount * 3 + 2] = ((extLeft[1] - extRight[1])/(extLeft[0] - extRight[0]) ) / 3.14 * 180
# #                         yellowBin[binCount * 3 + 2] = cv2.fitEllipse(yellow_c)[2]
#                         yellowBin[binCount * 3 + 2] = getOrientation(yellow_c, transformed_frame) / 3.14 * 180
#                      except Exception as e:
#                         yellowBin[binCount * 3 + 2] = 0
#                         print("No yellow detected")




                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame1, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame1, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text


            # Draw framerate in corner of frame
            cv2.putText(frame1,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            print("Trolley", trolley)
            print("Bin", yellowBin)
            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow("HSV",mask)
            cv2.imshow('Object detector', frame1)
            cv2.imshow('Transformed Frame', transformed_frame)
                        # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1
            print("Bin Count",binCount)
            sendValues(cvMode)
            trolleyCount = 0
            binCount = 0
            (yellowBin[0],yellowBin[1], yellowBin[2], yellowBin[3], yellowBin[4], yellowBin[5]) = (0,0,0, 0,0,0)
            (redDistance[0],redDistance[1]) = (greenDistance[0],greenDistance[1]) = (blueDistance[0], blueDistance[1]) = (0,0)
            (trolley[0],trolley[1], trolley[2], trolley[3], trolley[4], trolley[5]) = (0,0,0,0,0,0)
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
#     sendValues()
        elif cvMode == 5:

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()
            nearest_trolley_dist = float('inf')
            nearest_trolley_center = (0,0)

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            boxes, classes, scores = modelResult(frame)
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            objects[0], objects[3], objects[6], objects[9] = 0, 0, 0 , 0
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    cX = int((xmax+xmin)/2)
                    cY = int((ymax+ymin)/2)



                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.circle(frame, (cX,cY), 7, (255,255,255), -1)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index

                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'

                    if object_name == "Trolley":
                        dist_to_center = ((cX - imW/2)**2 + (cY - imH/2)**2)**0.5
                        if dist_to_center < nearest_trolley_dist:
                            nearest_trolley_dist = dist_to_center
                            nearest_trolley_center = (cX, cY)



                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text



            (line[0], line[1]) = nearest_trolley_center
           # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)
            print("Center X: ", line[0])
            print("Center Y: ", line[1])
            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1
            sendValues(cvMode)
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()
    videostream.stop()

if __name__ == '__main__':
    main()
