import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('CascadeClassifiers/faceCascade.xml')
eye_cascade = cv2.CascadeClassifier('CascadeClassifiers/eyeCascade.xml')

cap = cv2.VideoCapture(0)


def get_eyes(img): # returns the coordinates of two eyes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Find the biggest face present in the frame
    bigFace = (0, 0, 0, 0)
    bigFaceSize = 0
    for (x, y, w, h) in faces:
        currFaceSize = (x+w)*(y+h)
        if currFaceSize > bigFaceSize:
            bigFace = (x, y, w, h)

    # Find eye cascades in the biggest face
    fx, fy, fw, fh = bigFace[0], bigFace[1], bigFace[2], bigFace[3]
    roi_gray = gray[fy:fy + fh, fx:fx + fw]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    #Find the highest eyes within the face
    topEye1 = (0, 0, 0, 0)
    topEyeHeight1 = 1000000
    topEye2 = (0, 0, 0, 0)
    topEyeHeight2 = 1000000
    for (ex,ey,ew,eh) in eyes:
        if len(eyes) >= 2:  # Check that there are at least 2 eyes in frame
            if ey < topEyeHeight1:
                topEyeHeight2 = topEyeHeight1
                topEye2 = topEye1
                topEyeHeight1 = ey
                topEye1 = (ex, ey, ew, eh)
            elif ey < topEyeHeight2:
                topEyeHeight2 = ey
                topEye2 = (ex, ey, ew, eh)

    return ((topEye1[0]+(0.5*topEye1[2])), (topEye1[1]+(0.5*topEye1[3]))), ((topEye2[0]+(0.5*topEye2[2])), (topEye2[1]+(0.5*topEye2[3])))


ret, img = cap.read()
eye_pos1, eye_pos2 = get_eyes(img)

cap.release()
cv2.destroyAllWindows()