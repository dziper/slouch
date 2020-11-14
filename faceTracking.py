#Tutorial: https://www.guidodiepen.nl/2017/02/detecting-and-tracking-a-face-with-python-and-opencv/

#Import the OpenCV library
import cv2

#Initialize a face cascade using the frontal face haar cascade provided
#with the OpenCV2 library
faceCascade = cv2.CascadeClassifier('CascadeClassifiers/faceCascade.xml')

#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600

#Open the first webcame device
capture = cv2.VideoCapture(0)

#Create two opencv named windows
cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

#Position the windows next to eachother
cv2.moveWindow("base-image",0,100)
cv2.moveWindow("result-image",400,100)

#Start the window thread for the two windows we are using
cv2.startWindowThread()

rectangleColor = (0,165,255)


def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=3, search_width = 20):
    columns = img_gray[y:y+h, x-search_width/2:x+search_width/2];
    column_average = columns.mean(axis=1);
    gradient = np.gradient(column_average, 3);
    gradient = np.absolute(gradient); # abs gradient value
    max_indicies = np.argpartition(gradient, -top_values_to_consider)[-top_values_to_consider:] # indicies of the top 5 values
    max_values = gradient[max_indicies];
    if(max_values.sum() < top_values_to_consider): return None; # return none if no large gradient exists - probably no shoulder in the range
    weighted_indicies = (max_indicies * max_values);
    weighted_average_index = weighted_indicies.sum() / max_values.sum();
    index = int(weighted_average_index);
    index = y + index;
    return index;

def detect_shoulder(img_gray, face, direction, x_scale=0.75, y_scale=0.75):
    x_face, y_face, w_face, h_face = face; # define face components

    # define shoulder box componenets
    w = int(x_scale * w_face);
    h = int(y_scale * h_face);
    y = y_face + h_face * 3/4; # half way down head position
    if(direction == "right"): x = x_face + w_face - w / 20; # right end of the face box
    if(direction == "left"): x = x_face - w + w/20; # w to the left of the start of face box
    rectangle = (x, y, w, h);

    # calculate position of shoulder in each x strip
    x_positions = [];
    y_positions = [];
    for delta_x in range(w):
        this_x = x + delta_x;
        this_y = calculate_max_contrast_pixel(img_gray, this_x, y, h);
        if(this_y is None): continue; # dont add if no clear best value
        x_positions.append(this_x);
        y_positions.append(this_y);

    # extract line from positions
    #line = [(x_positions[5], y_positions[5]), (x_positions[-10], y_positions[-10])];
    lines = [];
    for index in range(len(x_positions)):
        lines.append((x_positions[index], y_positions[index]));

    # extract line of best fit from lines
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions,y_positions)
    line_y0 = int(x_positions[0] * slope + intercept)
    line_y1 = int(x_positions[-1] * slope + intercept);
    line = [(x_positions[0], line_y0), (x_positions[-1], line_y1)];

    # decide on value
    #value = intercept;
    value = np.array([line[0][1], line[1][1]]).mean();

    # return rectangle and positions
    return line, lines, rectangle, value;



while True:
    #Retrieve the latest image from the webcam
    rc,fullSizeBaseImage = capture.read()

    #Resize the image to 320x240
    baseImage = cv2.resize(fullSizeBaseImage, ( 320, 240))


    #Check if a key was pressed and if it was Q, then destroy all
    #opencv windows and exit the application
    pressedKey = cv2.waitKey(2)
    if pressedKey == ord('Q'):
        cv2.destroyAllWindows()
        exit(0)



    #Result image is the image we will show the user, which is a
    #combination of the original image from the webcam and the
    #overlayed rectangle for the largest face
    resultImage = baseImage.copy()


    #For the face detection, we need to make use of a gray colored
    #image so we will convert the baseImage to a gray-based image
    gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    #Now use the haar cascade detector to find all faces in the
    #image
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)


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
        cv2.rectangle(resultImage,  (bigFace[0]-10, bigFace[1]-20),(bigFace[0] + bigFace[2]+10 , bigFace[1] + bigFace[3]+20),rectangleColor,2)

    #Since we want to show something larger on the screen than the
    #original 320x240, we resize the image again
    #
    #Note that it would also be possible to keep the large version
    #of the baseimage and make the result image a copy of this large
    #base image and use the scaling factor to draw the rectangle
    #at the right coordinates.
    largeResult = cv2.resize(resultImage,(OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

    #Finally, we want to show the images on the screen
    cv2.imshow("base-image", baseImage)
    cv2.imshow("result-image", largeResult)
