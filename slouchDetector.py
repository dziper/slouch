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

# need this line or else get weird abort when you run another popup
# pymsgbox.alert("Welcome to slouchDetector9000", "Hey!")

SLOUCH_THRESH = 5
SLOUCH_TIMER = 5
MAX_COUNTOUR_SIZE = 800

ESC = 27

centerAngle = 54
angleBetween = 0
togglePause = False

getMinY = None

# HSV color bounds
colorBounds = ([5, 130, 150], [10, 190, 255])
croppedImage = None
croppedHSV = None

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", type=int, default=0, help="camera source")
args = vars(ap.parse_args())

#cv.imshow("Image", image)

stream = cv.VideoCapture(args["source"])
# boundaries for photo.png
name = "frames"
cv.namedWindow(name)

classifier = None

# set up mouse movement detection
mouseX = None
mouseY = None


def getMousePos(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_MOUSEMOVE:
        mouseX = x
        mouseY = y

refPt = []
cropping = False

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
		cv.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv.imshow("image", image)


def add_line_to_csv(filename, datalist):
    with open(filename, mode='a') as currFile:
        writer = csv.writer(currFile, delimiter=',', quotechar='"')
        writer.writerow(datalist)

cv.setMouseCallback(name, getMousePos)

(grabbed, frame) = stream.read()
cleanFrame = frame.copy()

print("frame shape: " + str(frame.shape))

key = -1

startTime = None
currTime = None

if not frame.size == 0:
    getMinY = frame[0,0,0]

while key != ESC:
    currTime = datetime.datetime.now()

    if not togglePause:
        (grabbed, frame) = stream.read()
        cleanFrame = frame.copy()

    if togglePause:
        frame = cleanFrame.copy()

    lower = np.array(colorBounds[0], dtype="uint8")
    upper = np.array(colorBounds[1], dtype="uint8")

    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsvFrame, lower, upper)
    # returns black and white mask
    maskedImg = cv.bitwise_and(frame, frame, mask=mask)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("mask", mask)

    frameData = []

    eyeLoc1, eyeLoc2 = get_eyes(frame)
    noEyeData = False
    if eyeLoc1 == (0,0) and eyeLoc2 == (0,0):
        noEyeData = True
    else:
        frameData.append(eyeLoc1)
        frameData.append(eyeLoc2)

    shoulderData = detectShoulders(gray, mask)
    noShoulderData = False
    if shoulderData is None:
        # print("No shoulder Data")
        noShoulderData = True
    else:
        right_line, right_slope, left_line, left_slope = shoulderData

        if not (np.isnan((right_slope,left_slope)).any() or np.isnan(right_line).any() or np.isnan(left_line).any()):
            right_beginning = (int(right_line[0,0]),int(right_line[0,1]))
            right_end = (int(right_line[-1,0]),int(right_line[-1,1]))
            frame = cv.line(frame, right_beginning, right_end, (0,255,0), 3)

            left_beginning = (int(left_line[0,0]),int(left_line[0,1]))
            left_end = (int(left_line[-1,0]),int(left_line[-1,1]))
            frame = cv.line(frame, left_beginning, left_end, (0,255,0), 3)

            # slopeDiff = right_slope - left_slope
            # angleBetween = np.arctan2(slopeDiff,1) * 180 / math.pi

            frameData += (right_beginning, right_end, right_slope)
            frameData += (left_beginning, left_end, left_slope)
        else:
            noShoulderData = True

    if not classifier is None:
        classifier.newData(frameData)
        slouchConfidence = classifier.classify()

        cv.putText(frame, "Confidence: " + str(np.round(slouchConfidence, decimals = 3)), (10, 100),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        circleColor = (0,255,0)
        if slouchConfidence > 0.5:
            circleColor = (0,0,255)
        cv.circle(frame, (30, 30), 10, circleColor, -1)

    if noShoulderData:
        # can't find contours
        startTime = None
        cv.putText(frame, "Can't find face/shoulders", (10, 25),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cursorBGR = [-1, -1, -1]
    if mouseX != None and mouseX < len(frame[0]):
        cursorHSV = hsvFrame[mouseY][mouseX]
        cursorBGR = frame[mouseY][mouseX]
    else:
        cv.putText(frame, "cant find mouse", (10, 660), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if cursorBGR[0] != -1:
        rgbString = "R: " + str(cursorBGR[2]) + " G: " + \
            str(cursorBGR[1]) + " B: " + str(cursorBGR[0])

        hsvString = "H: " + str(cursorHSV[0]) + " S: " + \
            str(cursorHSV[1]) + " V: " + str(cursorHSV[2])

        mouseString = "X: " + str(mouseX) + " Y: " + str(mouseY)

        cv.putText(frame, hsvString, (10, getMinY + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(frame, rgbString, (10, getMinY + 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)
        cv.putText(frame, mouseString, (10, getMinY + 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)
    # need to convert thresholded image to BGR or else hstack cant stack the images (color img has 3 channel, gray has 1)
    cv.imshow(name, frame)
    # cv.imshow(name, frame)

    if key == ord('c'):
        if len(frameData) == 8:
            if classifier is None:
                classifier = DataClassifier(frameData, frameData)
            else:
                classifier.newData(frameData, classify = True)
        # if user presses C
        print("calibrated classifier")


    if key == ord('p'):
        # if user presses P
        print("play/pause")
        togglePause = not togglePause

    if key == ord('s'):
        ds = str(currTime)
        cutString = ds[0:10] + "-" + ds[11:19]
        dateString = cutString.replace(":", "-")
        cleanFile = "dataImages/Clean/"+"Clean"+dateString+".jpg"
        overlayFile = "dataImages/Overlay/"+"Overlay"+dateString+".jpg"

        cv.imwrite(cleanFile, cleanFrame)
        print("Saving Clean Frame in " + cleanFile)
        cv.imwrite(overlayFile, frame)
        print("Saving Overlay Frame in " + overlayFile)

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
                break
        # if there are two reference points, then crop the region of interest
        # from the image and display it
        if len(refPt) == 2:
            croppedImage = cleanFrame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            # cv.imshow("ROI", roi)
            # cv.waitKey(0)

            print(refPt)

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
            print(minCroppedHSV)
            print(maxCroppedHSV)

    if startTime != None:
        if (currTime-startTime).total_seconds() > SLOUCH_TIMER:
            # pymsgbox.alert("Stop slouching!", "Hey!")
            # print("Slouch Timer!")
            pass

    key = cv.waitKey(1) & 0xFF


cv.destroyAllWindows()
