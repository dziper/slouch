#Tutorial: https://www.guidodiepen.nl/2017/02/detecting-and-tracking-a-face-with-python-and-opencv/

#Import the OpenCV library
import cv2 as cv
import numpy as np
from scipy import stats;
import math
import lineDetection

#Initialize a face cascade using the frontal face haar cascade provided
#with the Opencv library
faceCascade = cv.CascadeClassifier('CascadeClassifiers/faceCascade.xml')
colorBounds = ([170, 100, 15], [185, 180, 190])

def highestWhite(img_gray, x, minY = 0):
    x = int(x)
    minY = int(minY)
    column = img_gray[minY:,x]
    for i in range(len(column)):
        val = column[i]
        if val > 0:
            return i+minY
    return None

def detect_shoulder(img_gray, face, direction, x_scale=0.5, y_scale=0.75):
    x_face, y_face, w_face, h_face = face; # define face components

    # x_scale = 0.7

    HEIGHT_MULTIPLIER = 1

    # define shoulder box componenets
    w = int(x_scale * w_face);
    h = int(y_scale * h_face);
    y = y_face + h_face * HEIGHT_MULTIPLIER; # part way down head position
    if(direction == "right"): x = x_face + w_face; # right end of the face box
    if(direction == "left"): x = x_face - w; # w to the left of the start of face box
    rectangle = (x, y, w, h);

    # calculate position of shoulder in each x strip
    x_positions = np.array([])
    y_positions = np.array([])
    for delta_x in range(w):
        this_x = x + delta_x;
        # this_y = calculate_max_contrast_pixel(img_gray, this_x, y, h);
        this_y = highestWhite(img_gray, this_x, minY = y_face + h_face/2)
        if(this_y is None): continue; # dont add if no clear best value
        x_positions = np.append(x_positions, int(this_x))
        y_positions = np.append(y_positions, int(this_y))

    # extract line from positions
    #line = [(x_positions[5], y_positions[5]), (x_positions[-10], y_positions[-10])];
    if direction == "left":
        x_positions = np.flip(x_positions)
        y_positions = np.flip(y_positions)
    points = np.stack([x_positions,y_positions],axis = -1)

    # extract line of best fit from lines

    # slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions,y_positions)

    corrCoeff, line, slope, intercept = lineDetection.findBestFit(x_positions,y_positions,plot=False)

    return line, slope, points

def draw_line_onto_image(img, line, color="GREEN"):
    beginning = line[0];
    end = line[1];
    if(color=="GREEN"): color = (0, 255, 0);
    if(color=="BLUE"): color = (255, 0, 0);
    if(color=="RED"): color = (0, 0, 255);
    cv.line(img, beginning, end, color, 1)
    return img;

def plotPoints(img, pointList, color = (0,0,255)):
    for point in pointList:
        x = int(point[0])
        y = int(point[1])
        img[y,x] = color

def detectShoulders(img_gray, mask):
    #Now use the haar cascade detector to find all faces in the image
    faces = faceCascade.detectMultiScale(img_gray, 1.3, 5)

    maxArea = 0
    x = 0
    y = 0
    w = 0
    h = 0
    #Loop over all faces and check if the area for this face is
    #the largest so far
    bigFace = None
    largestArea = -1
    for face in faces:
        (_x,_y,_w,_h) = face
        if  _w*_h > maxArea:
            x = _x
            y = _y
            w = _w
            h = _h
            maxArea = w*h

        if maxArea > largestArea:
            largestArea = maxArea
            bigFace = face

    if largestArea > 0:
        # print("new frame")
        # cv.rectangle(img_gray,  (bigFace[0]-10, bigFace[1]-20),(bigFace[0] + bigFace[2]+10 , bigFace[1] + bigFace[3]+20),rectangleColor,2)
        lower = np.array(colorBounds[0], dtype="uint8")
        upper = np.array(colorBounds[1], dtype="uint8")

        right_line, right_slope, right_points = detect_shoulder(mask, bigFace, "right")
        left_line, left_slope, left_points = detect_shoulder(mask, bigFace, "left")

        return right_line, right_slope, left_line, left_slope

    return None


stream = cv.VideoCapture(0)

name = "frames"
cv.namedWindow(name)

while True:
    grabbed, frame = stream.read()
    key = cv.waitKey(1) & 0xFF

    lower = np.array(colorBounds[0], dtype="uint8")
    upper = np.array(colorBounds[1], dtype="uint8")

    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsvFrame, lower, upper)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow(name, mask)

cv.destroyAllWindows()
