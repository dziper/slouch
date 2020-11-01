#keep me open for autocomplete :)
import cv2
import imutils

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required = True, help = "path to input image")
# ap.add_argument("-o", "--output", required = True, help = "path to output image")
# args = vars(ap.parse_args())

image = cv2.imread("images/inception.jpg")
(h,w,d) = image.shape
#print dimenstions of image, depth is how many channels image has
cv2.imshow("Image", image)

(B,G,R) = image[100,50]
#Notice BGR not RGB. Also pixel coordinates are flipped [y,x] :(

roi = image[14:93, 363:432]
#manually extract region of interest
cv2.imshow("ROI", roi)
cv2.waitKey(0)

fixedResize = cv2.resize(image, (200,200))
cv2.imshow("Fixed Resize",fixedResize)
#doesn't conserve aspect ratio

#scaled resize
scale = 300
r = scale/w
dim =  (scale, int(h*r))
resized = cv2.resize(image, dim)
#orrr easy mode resize using imutils
ezResize = imutils.resize(image, width = 300)
cv2.imshow("Scaled resize",ezResize)
cv2.waitKey(0)

#rotate image
center = (w//2, h//2)
#// for integer division
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
#or easy rotate using imutils
ezRotated = imutils.rotate(image, -45)
#image is kinda clipped
boundRotated = imutils.rotate_bound(image, 45)
#for some reason rotate_bound() the degrees are inverse..?
cv2.imshow("Rotated",ezRotated)
cv2.waitKey(0)

#blur image to reduce noise
#kernel (11,11) must be odd ints
blurred = cv2.GaussianBlur(image, (11,11),0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

#Manually Drawing
#copy so you don't change the actual image
output = image.copy()
cv2.rectangle(output, (363,14), (432,93), (0,0,255), 3)
#cv2.rectangle(image, (x1,y1),(x2,y2), (b,g,r), line_width)
cv2.circle(output, (200,300), 20, (255,0,0), -1)
#make line_width negative to fill it
cv2.line(output, (100,100),(200,200), (0,255,0), 3)
cv2.putText(output, "Inceptionnn", (10,h-25), cv2.FONT_HERSHEY_DUPLEX, 2, (96,154,230), 5)
#cv2.putText(image, "text", (x,y), cv2.FONT, scale, (r,g,b), thickness)
cv2.imshow("Drawing",output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#detecting contours
tetris = cv2.imread("images/tetris_blocks.png")
cv2.imshow("Tetris", tetris)
gray = cv2.cvtColor(tetris, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 150)
#cv2.Canny(grayImg, minThresh, maxThresh, aperture_size = 3)
cv2.imshow("Canny edged" ,edged)

threshed = cv2.threshold(gray, 225,255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresholded", threshed)
cv2.waitKey(0)

cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#grab_contours() very important, fixes compatibility issue
out = tetris.copy()

#loop through each contour
for c in cnts:
    cv2.drawContours(out, [c], -1, (240, 0, 159), 3)
    #make sure convert each contour to a list [c]
    cv2.imshow("Contours", out)
    cv2.waitKey(0)

out2 = tetris.copy()
#or just draw all contours at once by putting in list
cv2.drawContours(out2, cnts, -1, (240, 0, 159), 3)
cv2.imshow("contours 2", out2)

text = "I found {} objects".format(len(cnts))
cv2.putText(out, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
cv2.imshow("Contours", out)

cv2.waitKey(0)

#erosion/dilation
mask = threshed.copy()
#remove pixels
mask = cv2.erode(mask, None, iterations = 5)
cv2.imshow("Eroded", mask)

mask2 = threshed.copy()
#add pixels
mask2 = cv2.dilate(mask2, None, iterations = 5)
cv2.imshow("Dilated", mask2)

mask = threshed.copy()
#mask away parts of the image we don't care about
output = cv2.bitwise_and(tetris,tetris, mask = mask)
cv2.imshow("Out", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
