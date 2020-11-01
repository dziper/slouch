import cv2
import argparse
import imutils
import numpy as np
import datetime
import pymsgbox

# need this line or else get weird abort when you run another popup
pymsgbox.alert("Welcome to slouchDetector9000", "Hey!")

ESC = 27
centerAngle = 54
slouchThresh = 5
slouchTimer = 5
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", type=int, default=0, help="camera source")
args = vars(ap.parse_args())

#cv2.imshow("Image", image)

stream = cv2.VideoCapture(args["source"])
# boundaries for photo.png
name = "frames"
cv2.namedWindow(name)

# set up mouse movement detection
mouseX = None
mouseY = None


def getMousePos(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_MOUSEMOVE:
        mouseX = x
        mouseY = y


cv2.setMouseCallback(name, getMousePos)


greenBounds = ([80, 60, 60], [150, 150, 150])

key = -1

startTime = None
currTime = None

while key != ESC:
    currTime = datetime.datetime.now()
    (grabbed, frame) = stream.read()
    lower = np.array(greenBounds[0], dtype="uint8")
    upper = np.array(greenBounds[1], dtype="uint8")

    mask = cv2.inRange(frame, lower, upper)
    # returns black and white mask
    greenImg = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(greenImg, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshed = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    contAngles = []

    for cont in contours:
        M = cv2.moments(cont)
        if M["m00"] == 0:
            continue

        extLeft = tuple(cont[cont[:, :, 0].argmin()][0])
        extRight = tuple(cont[cont[:, :, 0].argmax()][0])
        extTop = tuple(cont[cont[:, :, 1].argmin()][0])
        extBottom = tuple(cont[cont[:, :, 1].argmax()][0])

        contWidth = extRight[0] - extLeft[0]
        contHeight = extBottom[1] - extTop[1]
        contArea = contWidth * contHeight

        contArea = cv2.contourArea(cont)

        rect = cv2.minAreaRect(cont)
        box1 = cv2.boxPoints(rect)
        box = np.int0(box1)
        boxArea = cv2.contourArea(box)

        if boxArea < 1500:
            continue

        angle = rect[2]
        contAngles.append(angle)

        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        cv2.circle(frame, (cX, cY), 3, (0, 0, 255), -1)
        cv2.drawContours(frame, [cont], -1, (255, 255, 255), 2)
        cv2.drawContours(frame, [box], -1, (0, 255, 255), 2)

    if(len(contAngles) > 1):
        # can see two contours
        angleBetween = np.abs(contAngles[0] - contAngles[1])
        distFromCenter = np.abs(angleBetween - centerAngle)
        if (distFromCenter > slouchThresh):
            # detected slouch
            if startTime == None:
                startTime = currTime
            cv2.circle(frame, (30, 30), 10, (0, 150 - distFromCenter, 150 + distFromCenter), -1)
            cv2.putText(frame, "Stop slouching!", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            startTime = None
            cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1)

    else:
        # can't find contours
        startTime = None
        cv2.putText(frame, "Can't find contours!", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cursorBGR = [-1, -1, -1]
    if mouseX != None and mouseX < len(frame[0]):
        cursorBGR = frame[mouseY][mouseX]

    if cursorBGR[0] != -1:
        colorString = "R: " + str(cursorBGR[2]) + " G: " + \
            str(cursorBGR[1]) + " B: " + str(cursorBGR[0])
        cv2.putText(frame, colorString, (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # need to convert thresholded image to BGR or else hstack cant stack the images (color img has 3 channel, gray has 1)
    cv2.imshow(name, np.hstack([frame, cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)]))
    # cv2.imshow(name, frame)

    if key == 99:
        # if user presses C
        print("calibrated current angle")
        centerAngle = angleBetween

    if startTime != None:
        if (currTime-startTime).total_seconds() > slouchTimer:
            pymsgbox.alert("Stop slouching!", "Hey!")

    key = cv2.waitKey(1) & 0xFF


cv2.destroyAllWindows()
