#Tutorial: https://www.guidodiepen.nl/2017/02/detecting-and-tracking-a-face-with-python-and-opencv/

#Import the OpenCV library
import cv2
import numpy as np
from scipy import stats;
import math
import lineDetection

#Initialize a face cascade using the frontal face haar cascade provided
#with the OpenCV2 library
faceCascade = cv2.CascadeClassifier('CascadeClassifiers/faceCascade.xml')
colorBounds = ([170, 100, 15], [180, 180, 100])

def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=3, search_width = 20):
    columns = img_gray[int(y):int(y+h), int(x-search_width/2):int(x+search_width/2)];

    column_average = columns.mean(axis=1);
    gradient = np.gradient(column_average, 3);
    gradient = np.absolute(gradient); # abs gradient value
    max_indicies = np.argpartition(gradient, -top_values_to_consider)[-top_values_to_consider:] # indicies of the top 5 values
    max_values = gradient[max_indicies];
    if(max_values.sum() < top_values_to_consider): return None; # return none if no large gradient exists - probably no shoulder in the range
    weighted_indicies = (max_indicies * max_values);
    weighted_average_index = weighted_indicies.sum() / max_values.sum();

    if math.isnan(weighted_average_index):
        return None

    index = int(weighted_average_index);
    index = y + index;
    return index;

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
    cv2.line(img, beginning, end, color, 1)
    return img;

def plotPoints(img, pointList, color = (0,0,255)):
    for point in pointList:
        x = int(point[0])
        y = int(point[1])
        img[y,x] = color


def detectShoulders(img_gray, mask):
    #Now use the haar cascade detector to find all faces in the image
    faces = faceCascade.detectMultiScale(img_gray, 1.3, 5)

    #For now, we are only interested in the 'largest' face, and we
    #determine this based on the largest area of the found
    #rectangle. First initialize the required variables to 0
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

        #If one or more faces are found, draw a rectangle around the
        #largest face present in the picture
        if maxArea > largestArea:
            largestArea = maxArea
            bigFace = face


    if largestArea > 0:
        # print("new frame")

        # cv2.rectangle(img_gray,  (bigFace[0]-10, bigFace[1]-20),(bigFace[0] + bigFace[2]+10 , bigFace[1] + bigFace[3]+20),rectangleColor,2)



        lower = np.array(colorBounds[0], dtype="uint8")
        upper = np.array(colorBounds[1], dtype="uint8")

        right_line, right_slope, right_points = detect_shoulder(mask, bigFace, "right")
        left_line, left_slope, left_points = detect_shoulder(mask, bigFace, "left")

        return right_line, right_slope, left_line, left_slope

    return None
