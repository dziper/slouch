import cv2 as cv
import argparse
import imutils
import numpy as np
import datetime
import pymsgbox
import copy
from eyeDetector import get_eyes
from shoulderTracking import detectShoulders
from dataClassifier import DataClassifier
import csv
import math


ESC = 27

# Intro!
pymsgbox.alert("Welcome to our Super Spicy Slouch Detector! \nPlease take a look at the README file before proceeding.")

def click_n_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
    cv.imshow("image", frame)

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", type=int, default=0, help="camera source")
args = vars(ap.parse_args())

stream = cv.VideoCapture(args['source'])
if not (stream.isOpened()):
    print("Could not open video device")
    quit()

# Set resolution of frame
stream.set(3, 640)
stream.set(4, 480)

# Threshold boundaries
# HSV color bounds
colorBounds = None
key = -1

togglePause = False

getMinY = None

# Set up image cropping
croppedImage = None
croppedHSV = None
refPt = []
cropping = False

slider_scale = 0.5
add_to_x = 0

# Set up window
name = "frame"
cv.namedWindow(name)

# Init classifier
classifier = None

# set up mouse movement detection
mouseX = None
mouseY = None

# Detect mouse position
def getMousePos(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_MOUSEMOVE:
        mouseX = x
        mouseY = y
cv.setMouseCallback(name, getMousePos)

# Create two trackbars to allow adjustments
def make_trackbar_outside(window):
    cv.createTrackbar("Slide for shoulder distance", window, 10, 20, lambda x: x)

def make_trackbar_inside(window):
    cv.createTrackbar("Slide for distance from neck", window, 0, 20, lambda x: x)

# Show the points being used for linear regression
def plot_points(img, point_list):
    for (x,y) in point_list:
        int_x = int(x)
        int_y = int(y)
        image = cv.circle(img, (int_x,int_y), radius=1, color=(0,255,255), thickness=-1)

# Calculate the median of data for more stable data
def medianOfData(dataHist):
    dataHist = np.array(dataHist)
    outData = []
    for i in range(dataHist.shape[1]):
        dataColumn = dataHist[:,i]
        testEl = dataColumn[0]

        # Some data points are ints, some are tuples, account for both cases
        if type(testEl) != type((1,1)):
            outData.append(np.median(dataColumn))
        else:
            xs = []
            ys = []
            for i in range(len(dataColumn)):
                xs.append(dataColumn[i][0])
                ys.append(dataColumn[i][1])
            outData.append((np.median(xs), np.median(ys)))

    return outData

(grabbed, frame) = stream.read()
if frame is not None:
    cleanFrame = frame.copy()
    if not frame.size == 0:
        getMinY = frame[0,0,0]

# Set up timer
startTime = 0
currTime = None
circleColor = None
SLOUCH_TIMER = 50

# Set up data history
dataHistory = []
maxHistoryLength = 30

calibrated = False

# Control loop to calibrate the initial colorBounds
while(not calibrated):
    key = cv.waitKey(1) & 0xFF

    ret, frame = stream.read()

    if frame is None:
        print("skipped frame")
        continue

    cv.imshow(name,frame)

    if key == ord('x'):
        image = frame.copy()
        cv.namedWindow("image")
        cv.setMouseCallback("image", click_n_crop)

        cancelCrop = False
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv.imshow("image", frame)
            key = cv.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                frame = image.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                cv.destroyWindow("image")
                break
            elif key == ESC:
                cancelCrop = True
                break
        if cancelCrop:
            break

        # if there are two reference points, then crop the region of interest
        if len(refPt) == 2:
            croppedImage = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]


            croppedHSV = cv.cvtColor(croppedImage, cv.COLOR_BGR2HSV)
            minCroppedHSV = list(croppedHSV[0,0])
            maxCroppedHSV = list(croppedHSV[0,0])

            hues = np.array([])

            totalHSV = 0
            for i in range(0, croppedHSV.shape[0]):
                for j in range(0, croppedHSV.shape[1]):
                    for k in range(3):
                        val = croppedHSV[i,j][k] #loop through h,s,v
                        if k == 0:
                            hues = np.append(hues, val)
                        if val < minCroppedHSV[k]:
                            minCroppedHSV[k] = val
                        if val > maxCroppedHSV[k]:
                            maxCroppedHSV[k] = val
            # Find the median of the hue, and the max and mins of sat and val
            # Set colorBounds accourdingly
            medHue = np.median(hues)
            minCroppedHSV[0] = medHue - 5
            maxCroppedHSV[0] = medHue + 5
            colorBounds = (tuple(minCroppedHSV), tuple(maxCroppedHSV))
            calibrated = True

while key != ESC:
    currTime = datetime.datetime.now()

    if not togglePause:
        (grabbed, frame) = stream.read()
        if frame is None:
            print("skipped frame")
            continue
        cleanFrame = frame.copy()

    if togglePause:
        frame = cleanFrame.copy()

    make_trackbar_outside(name)
    make_trackbar_inside(name)

    # Threshold image using colorBounds
    lower = np.array(colorBounds[0], dtype="uint8")
    upper = np.array(colorBounds[1], dtype="uint8")
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsvFrame, lower, upper)
    # returns black and white mask
    maskedImg = cv.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale for Haar Classifiers (face detection)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("mask", mask)

    frameData = []
    # Get eye positions
    eyeLoc1, eyeLoc2 = get_eyes(frame)
    noEyeData = False
    if eyeLoc1 == (0,0) and eyeLoc2 == (0,0):
        noEyeData = True
    else:
        frameData.append(eyeLoc1)
        frameData.append(eyeLoc2)

    # Get shoulder positions
    curr_x_scale = cv.getTrackbarPos("Slide for shoulder distance", name)
    curr_add_to_x = cv.getTrackbarPos("Slide for distance from neck", name)
    shoulderData = detectShoulders(gray, mask, curr_x_scale, curr_add_to_x)
    noShoulderData = False
    if shoulderData is None:
        # print("No shoulder Data")
        noShoulderData = True
    else:
        right_line, right_slope, left_line, left_slope, right_points, left_points = shoulderData

        if not (np.isnan((right_slope,left_slope)).any() or np.isnan(right_line).any() or np.isnan(left_line).any()):
            # Extract necessary components for classification from shoulderData
            right_beginning = (int(right_line[0,0]),int(right_line[0,1]))
            right_end = (int(right_line[-1,0]),int(right_line[-1,1]))
            frame = cv.line(frame, right_beginning, right_end, (0,255,0), 3)

            left_beginning = (int(left_line[0,0]),int(left_line[0,1]))
            left_end = (int(left_line[-1,0]),int(left_line[-1,1]))
            frame = cv.line(frame, left_beginning, left_end, (0,255,0), 3)

            plot_points(frame, right_points)
            plot_points(frame, left_points)

            frameData += (right_beginning, right_end, right_slope)
            frameData += (left_beginning, left_end, left_slope)
        else:
            noShoulderData = True

    # Make sure frameData is valid
    if len(frameData) == 8:
        dataHistory += [frameData]
        if len(dataHistory) > maxHistoryLength:
            dataHistory.pop(0)

    if not classifier is None:
        # Classify using median
        rollingMedian = medianOfData(dataHistory)
        classifier.newData(rollingMedian)
        slouchConfidence = classifier.classify()

        # Display Slouch Level (confidence level)
        if not noShoulderData:
            cv.putText(frame, "Slouch Level: " + str(np.round(slouchConfidence, decimals = 3)), (50, 38),
                        cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Plot shoulder data points
        plot_points(frame, right_points)
        plot_points(frame, left_points)

        # Display colored slouch indicator
        circleColor = (0,255,0)
        if slouchConfidence > 0.5:
            circleColor = (0,0,255)
        cv.circle(frame, (30, 30), 10, circleColor, -1)

        # Display slouch meter
        cv.putText(frame, "Slouch Meter:", (20, frame.shape[0] - 17), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv.line(frame, (200,frame.shape[0] - 20), (600,frame.shape[0] - 20), (255, 255, 255), 3)
        if not (startTime is None):
            cv.line(frame, (200,frame.shape[0] - 20), (200 + int(400 * (startTime+1)/SLOUCH_TIMER),frame.shape[0] - 20), (0, 0, 255), 3)

    if noShoulderData:
        # can't find face/shoulders
        startTime = startTime - 1
        cv.putText(frame, "Can't find Face/Shoulders", (50, 38),
                    cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


    # Show the main image
    cv.imshow(name, frame)

    # Calibrate the classifier
    if key == ord('c'):
        # Make sure we have enough data to calibrate reliably
        if len(dataHistory) >= maxHistoryLength:
            calibratedMed = medianOfData(dataHistory)

            if len(frameData) == 8:
                if classifier is None:
                    classifier = DataClassifier(calibratedMed, calibratedMed)
                else:
                    classifier.newData(calibratedMed, calibrate = True)
                print("calibrated classifier")
        else:
            print("need more data, try again in a few seconds")

        # if user presses C

    # pause the feed
    if key == ord('p'):
        # if user presses P
        print("play/pause")
        togglePause = not togglePause

    # Crop a new color
    if key == ord('x'):
        image = cleanFrame.copy()
        cv.namedWindow("image")
        cv.setMouseCallback("image", click_n_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv.imshow("image", image)
            key = cv.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = cleanFrame.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                cv.destroyWindow("image")
                break
        # if there are two reference points, then crop the region of interest
        # from the image and display it
        if len(refPt) == 2:
            croppedImage = cleanFrame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            # cv.imshow("ROI", roi)
            # cv.waitKey(0)

            croppedHSV = cv.cvtColor(croppedImage, cv.COLOR_BGR2HSV)
            minCroppedHSV = list(croppedHSV[0,0])
            maxCroppedHSV = list(croppedHSV[0,0])

            hues = np.array([])

            totalHSV = 0
            for i in range(0, croppedHSV.shape[0]):
                for j in range(0, croppedHSV.shape[1]):
                    for k in range(3):
                        val = croppedHSV[i,j][k] #loop through h,s,v
                        if k == 0:
                            hues = np.append(hues, val)
                        if val < minCroppedHSV[k]:
                            minCroppedHSV[k] = val
                        if val > maxCroppedHSV[k]:
                            maxCroppedHSV[k] = val


            # avgCroppedHSV = (minCroppedHSV + maxCroppedHSV) / 2

            # print(croppedHSV.size)
            medHue = np.median(hues)
            minCroppedHSV[0] = medHue - 5
            maxCroppedHSV[0] = medHue + 5
            colorBounds = (tuple(minCroppedHSV), tuple(maxCroppedHSV))

    # Slouch alerts
    if circleColor == (0,255,0):
        startTime = 0
    elif circleColor == (0,0,255) and startTime != None and not togglePause:
        startTime = startTime + 1
        if startTime >= SLOUCH_TIMER:
            pymsgbox.alert("Hey you! Stop slouching!")
            startTime = 0
            pass

    key = cv.waitKey(1) & 0xFF

cv.destroyAllWindows()
print("Thanks for using Slouch Detector!")
