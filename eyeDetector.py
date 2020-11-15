import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('CascadeClassifiers/faceCascade.xml')
eye_cascade = cv2.CascadeClassifier('CascadeClassifiers/eyeCascade.xml')

cap = cv2.VideoCapture(0)


def get_eyes():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    bigFace = (0, 0, 0, 0)
    bigFaceSize = 0
    for (x, y, w, h) in faces:
        currFaceSize = (x+w)*(y+h)
        if currFaceSize > bigFaceSize:
            bigFace = (x, y, w, h)

    fx, fy, fw, fh = bigFace[0], bigFace[1], bigFace[2], bigFace[3]
    cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (255,0,0), 2)
    roi_gray = gray[fy:fy + fh, fx:fx + fw]
    roi_color = img[fy:fy+fh, fx:fx+fw]
    eyes = eye_cascade.detectMultiScale(roi_gray)

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


        # if ey < topEyeHeight1:
        #     topEyeHeight2 = topEyeHeight1
        #     topEye2 = topEye1
        #     topEyeHeight1 = ey
        #     topEye1 = (ex, ey, ew, eh)
        # elif ey < topEyeHeight2:
        #     topEyeHeight2 = ey
        #     topEye2 = (ex, ey, ew, eh)

    if topEye1 and topEye2:
        cv2.rectangle(roi_color, (topEye1[0], topEye1[2]), (topEye1[0] + topEye1[1], topEye1[2] + topEye1[3]), (0, 255, 0), 2)
        cv2.rectangle(roi_color, (topEye2[0], topEye2[2]), (topEye2[0] + topEye2[1], topEye2[2] + topEye2[3]), (0, 255, 0), 2)



        # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # eyesLocString = "X:" + str(ex + (ew / 2)) + " Y:" + str(ey + (eh / 2))
        # cv2.putText(img, eyesLocString, (10, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('img', img)
    return (topEye1[0]+(0.5*topEye1[2])), (topEye1[1]+(0.5*topEye1[3])), (topEye2[0]+(0.5*topEye2[2])), (topEye2[1]+(0.5*topEye2[3]))
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break


while 1:
    eye_posX1, eye_posY1, eye_posX2, eye_posY2 = get_eyes()
    print("x1:", eye_posX1, "y1:", eye_posY1)
    print("x2:", eye_posX2, "y2:", eye_posY2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# while 1:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#             eyesLocString = "X:" + str(ex + (ew / 2)) + " Y:" + str(ey + (eh / 2))
#             cv2.putText(img, eyesLocString, (10, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#
#
#
#     cv2.imshow('img',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
cap.release()
cv2.destroyAllWindows()