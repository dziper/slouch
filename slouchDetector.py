import cv2 as cv
import argparse
import imutils
import numpy as np
import datetime
import pymsgbox
import copy
from eyeDetector import get_eyes
from shoulderTracking import detectShoulders
import csv

# need this line or else get weird abort when you run another popup
# pymsgbox.alert("Welcome to slouchDetector9000", "Hey!")

SLOUCH_THRESH = 5
SLOUCH_TIMER = 5
MAX_COUNTOUR_SIZE = 800

ESC = 27

centerAngle = 54
togglePause = False

getMinY = None

# HSV color bounds
colorBounds = ([170, 90, 15], [185, 190, 200])

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", type=int, default=0, help="camera source")
args = vars(ap.parse_args())

#cv.imshow("Image", image)

stream = cv.VideoCapture(args["source"])
# boundaries for photo.png
name = "frames"
cv.namedWindow(name)

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
cleanFrame = copy.deepcopy(frame)

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
        cleanFrame = copy.deepcopy(frame)

    if togglePause:
        frame = copy.deepcopy(cleanFrame)

    lower = np.array(colorBounds[0], dtype="uint8")
    upper = np.array(colorBounds[1], dtype="uint8")

    # eyeLoc1, eyeLoc2 = get_eyes(frame)

    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsvFrame, lower, upper)
    # returns black and white mask
    maskedImg = cv.bitwise_and(frame, frame, mask=mask)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("mask", mask)

    shoulderData = detectShoulders(gray, mask)
    noShoulderData = False
    if shoulderData is None:
        print("No shoulder Data")
        noShoulderData = True
    else:
        right_line, right_slope, left_line, left_slope = shoulderData
        print("slope diff")
        right_beginning = (int(right_line[0,0]),int(right_line[0,1]))
        right_end = (int(right_line[-1,0]),int(right_line[-1,1]))
        frame = cv.line(frame, right_beginning, right_end, (0,255,0), 3)

        left_beginning = (int(left_line[0,0]),int(left_line[0,1]))
        left_end = (int(left_line[-1,0]),int(left_line[-1,1]))
        frame = cv.line(frame, left_beginning, left_end, (0,255,0), 3)

        print(right_slope - left_slope)
        # print(left_slope)

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
        # if user presses C
        print("calibrated current angle")
        centerAngle = angleBetween

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
        togglePause = not togglePause
        cv.imwrite('image.jpg', frame)
        image = cv.imread('image.jpg')
        clone = image.copy()
        cv.namedWindow("image")
        cv.setMouseCallback("image", click_n_crop)

        # keep looping until the 'q' key is pressed
        while True:
        	# display the image and wait for a keypress
        	cv.imshow("image", image)
        	key = cv.waitKey(1) & 0xFF
        	# if the 'r' key is pressed, reset the cropping region
        	if key == ord("r"):
        		image = clone.copy()
        	# if the 'c' key is pressed, break from the loop
        	elif key == ord("c"):
        		break
        # if there are two reference points, then crop the region of interest
        # from the image and display it
        if len(refPt) == 2:
        	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        	cv.imshow("ROI", roi)
        	cv.waitKey(0)

    if startTime != None:
        if (currTime-startTime).total_seconds() > SLOUCH_TIMER:
            # pymsgbox.alert("Stop slouching!", "Hey!")
            pass

    key = cv.waitKey(1) & 0xFF


cv.destroyAllWindows()
