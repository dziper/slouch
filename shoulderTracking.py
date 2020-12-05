import cv2 as cv
import numpy as np
from scipy import stats
import math
import lineDetection

#Initialize a face cascade using the frontal face haar cascade provided
#with the Opencv library
faceCascade = cv.CascadeClassifier('CascadeClassifiers/faceCascade.xml')
colorBounds = ([170, 90, 15], [185, 190, 200])

def highestWhite(gray, x, minY = 0):
    if x >= gray.shape[1] or x < 0 or minY >= gray.shape[0] or minY < 0:
        return None
    x = int(x)
    minY = int(minY)
    column = gray[minY:,x]
    for i in range(len(column)):
        val = column[i]
        if val > 0:
            return i+minY
    return None

def detect_shoulder(gray, face, direction, raw_add_to_x, raw_x_scale, y_scale=0.75):
    x_face, y_face, w_face, h_face = face; # define face components
    x_scale = raw_x_scale/20
    add_to_x = raw_add_to_x * 10

    HEIGHT_MULTIPLIER = 1

    # define shoulder box componenets
    w = int(x_scale * w_face)
    h = int(y_scale * h_face)
    y = y_face + h_face * HEIGHT_MULTIPLIER; # part way down head position
    if(direction == "right"): x = x_face + w_face + add_to_x; # right end of the face box
    if(direction == "left"): x = x_face - w - add_to_x; # w to the left of the start of face box
    rectangle = (x, y, w, h)

    # Find the shoulder for each x value by using the thresholded image
    x_positions = np.array([])
    y_positions = np.array([])
    for delta_x in range(w):
        this_x = x + delta_x
        # Get the highest white value in each x slice
        this_y = highestWhite(gray, this_x, minY = y_face + h_face/2)
        if(this_y is None): continue; # Skip x slice if necessary
        x_positions = np.append(x_positions, int(this_x))
        y_positions = np.append(y_positions, int(this_y))

    # extract line from positions
    if direction == "left":
        x_positions = np.flip(x_positions)
        y_positions = np.flip(y_positions)
    points = np.stack([x_positions,y_positions],axis = -1)

    # Use custom line detection to get the best line from points
    shoulderData = lineDetection.findBestFit(x_positions,y_positions,plot=False,low = 0.1)
    if shoulderData is None:
        return None

    corrCoeff, line, slope, intercept = shoulderData

    return line, slope, points

def plotPoints(img, pointList, color = (0,0,255)):
    for point in pointList:
        x = int(point[0])
        y = int(point[1])
        img[y,x] = color

def detectShoulders(gray, mask, scale, add_to_x):
    #Use Haar Cascade Classifier to find all faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    maxArea = 0
    x = 0
    y = 0
    w = 0
    h = 0

    # Find the largest face
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
        # Use the face to find data points of the right and left shoulders
        rightShoulderData = detect_shoulder(mask, bigFace, "right", add_to_x, scale)
        leftShoulderData = detect_shoulder(mask, bigFace, "left", add_to_x, scale)

        if rightShoulderData is None or leftShoulderData is None:
            return None

        right_line, right_slope, right_points = rightShoulderData
        left_line, left_slope, left_points = leftShoulderData
        return right_line, right_slope, left_line, left_slope, right_points, left_points
    else:
        pass
        # No face detected
    return None
