import cv2 as cv
import argparse
import imutils
import numpy as np
import datetime
import pymsgbox

# need this line or else get weird abort when you run another popup
# pymsgbox.alert("Welcome to slouchDetector9000", "Hey!")

ESC = 27
centerAngle = 54
slouchThresh = 5
slouchTimer = 5
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


cv.setMouseCallback(name, getMousePos)

(grabbed, frame) = stream.read()
print("frame shape: " + str(frame.shape))
# HSV color bounds
colorBounds = ([0, 100, 200], [10, 255, 255])

key = -1

startTime = None
currTime = None

while key != ESC:
    currTime = datetime.datetime.now()
    (grabbed, frame) = stream.read()
    lower = np.array(colorBounds[0], dtype="uint8")
    upper = np.array(colorBounds[1], dtype="uint8")

    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsvFrame, lower, upper)
    # returns black and white mask
    maskedImg = cv.bitwise_and(frame, frame, mask=mask)
    gray = cv.cvtColor(maskedImg, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    threshed = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]

    contours = cv.findContours(threshed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    contAngles = []

    for cont in contours:
        M = cv.moments(cont)
        if M["m00"] == 0:
            continue

        extLeft = tuple(cont[cont[:, :, 0].argmin()][0])
        extRight = tuple(cont[cont[:, :, 0].argmax()][0])
        extTop = tuple(cont[cont[:, :, 1].argmin()][0])
        extBottom = tuple(cont[cont[:, :, 1].argmax()][0])

        contWidth = extRight[0] - extLeft[0]
        contHeight = extBottom[1] - extTop[1]
        contArea = contWidth * contHeight

        contArea = cv.contourArea(cont)

        rect = cv.minAreaRect(cont)
        box1 = cv.boxPoints(rect)
        box = np.int0(box1)
        boxArea = cv.contourArea(box)

        if boxArea < 800:
            continue

        angle = rect[2]
        contAngles.append(angle)

        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        cv.circle(frame, (cX, cY), 3, (0, 0, 255), -1)
        cv.drawContours(frame, [cont], -1, (255, 255, 255), 2)
        cv.drawContours(frame, [box], -1, (0, 255, 255), 2)

    if(len(contAngles) > 1):
        # can see two contours
        angleBetween = np.abs(contAngles[0] - contAngles[1])
        distFromCenter = np.abs(angleBetween - centerAngle)

        cv.putText(frame, "Angle: " + str(np.round(distFromCenter, decimals = 3)), (10, 100),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        if (distFromCenter > slouchThresh):
            # detected slouch
            if startTime == None:
                startTime = currTime
            cv.circle(frame, (30, 30), 10, (0, 150 - distFromCenter, 150 + distFromCenter), -1)
            cv.putText(frame, "Stop slouching!", (10, 25),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            startTime = None
            cv.circle(frame, (30, 30), 10, (0, 255, 0), -1)

    else:
        # can't find contours
        startTime = None
        cv.putText(frame, "Can't find contours!", (10, 25),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cursorBGR = [-1, -1, -1]
    if mouseX != None and mouseX < len(frame[0]):
        cursorHSV = hsvFrame[mouseY][mouseX]
        cursorBGR = frame[mouseY][mouseX]

    if cursorBGR[0] != -1:
        rgbString = "R: " + str(cursorBGR[2]) + " G: " + \
            str(cursorBGR[1]) + " B: " + str(cursorBGR[0])

        hsvString = "H: " + str(cursorHSV[0]) + " S: " + \
            str(cursorHSV[1]) + " V: " + str(cursorHSV[2])
        # hsvString = "H: " + str(cursorHSV[0]) + " S: " + \ str(cursorBGR[1]) + " B: " + str(cursorBGR[0])
        cv.putText(frame, hsvString, (10, 680), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # need to convert thresholded image to BGR or else hstack cant stack the images (color img has 3 channel, gray has 1)
    cv.imshow(name, np.hstack([frame, cv.cvtColor(threshed, cv.COLOR_GRAY2BGR)]))
    # cv.imshow(name, frame)

    if key == 99:
        # if user presses C
        print("calibrated current angle")
        centerAngle = angleBetween

    if startTime != None:
        if (currTime-startTime).total_seconds() > slouchTimer:
            # pymsgbox.alert("Stop slouching!", "Hey!")
            pass

    key = cv.waitKey(1) & 0xFF


cv.destroyAllWindows()
