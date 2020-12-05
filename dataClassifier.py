import math
import numpy as np

class DataClassifier:
    # Initialize the classifier with calibrated Data
    def __init__(self,frameData, calibratedData):
        if len(frameData) != 8 or len(calibratedData) != 8:
            print("classifier is NONE")
        self.right_eye = [frameData[0], calibratedData[0]]
        self.left_eye = [frameData[1], calibratedData[1]]
        self.right_beginning = [frameData[2], calibratedData[2]]
        self.right_end = [frameData[3], calibratedData[3]]
        self.right_slope = [frameData[4], calibratedData[4]]
        self.left_beginning = [frameData[5], calibratedData[5]]
        self.left_end = [frameData[6], calibratedData[6]]
        self.left_slope = [frameData[7], calibratedData[7]]

    #update data (either data of each frame or )
    def newData(self, data, calibrate = False):
        if len(data) != 8:
            return False

        self.right_eye[calibrate] = data[0]
        self.left_eye[calibrate] = data[1]
        self.right_beginning[calibrate] = data[2]
        self.right_end[calibrate] = data[3]
        self.right_slope[calibrate] = data[4]
        self.left_beginning[calibrate] = data[5]
        self.left_end[calibrate] = data[6]
        self.left_slope[calibrate] = data[7]

        return True

    def classify(self):
        weightedSum = 0

        angleWeight = 25
        angleDiffVal = np.abs(self.getAngleDifference()/180 * angleWeight)

        ratioDifference = self.getEyeShoulderHeightRatio(calibrate = True)/self.getEyeShoulderHeightRatio()
        invRatioDifference = 1/ratioDifference
        ratioDifference = max(ratioDifference,invRatioDifference) - 1
        ratioWeight = 3
        ratioVal = np.abs(ratioDifference) * ratioWeight

        eyeAngleWeight = 10
        eyeAngleDiffVal = np.abs(self.getEyeAngleDifference()/180 * eyeAngleWeight)

        weightedSum += angleDiffVal + ratioVal + eyeAngleDiffVal

        # TODO: Add Eye Angle, eye width vs shoulder to eye Y dist ratio

        return (DataClassifier.sigmoid(weightedSum) - 0.5)*2

    #returns angle difference between calib and frame angle in degrees
    def getAngleDifference(self):
        calibSlopeDiff = self.right_slope[1] - self.left_slope[1]
        calibAngleBetween = np.arctan2(calibSlopeDiff,1) * 180 / math.pi

        slopeDiff = self.right_slope[0] - self.left_slope[0]
        angleBetween = np.arctan2(slopeDiff,1) * 180 / math.pi

        return calibAngleBetween - angleBetween

    def getEyeAngle(self, calibrate = False):
        slope = (self.right_eye[calibrate][1] - self.left_eye[calibrate][1])/ (self.right_eye[calibrate][0] - self.left_eye[calibrate][0])
        return np.arctan2(slope,1) * 180 / math.pi

    def getEyeAngleDifference(self):
        return self.getEyeAngle(calibrate = True) - self.getEyeAngle()

    # returns ratio beterrn eyes and shoulders
    def getEyeShoulderRatio(self, calibrate=False):
        eyeWidth = self.right_eye[calibrate][0] - self.left_eye[calibrate][0]

        # TODO: Check if left/right beginninin or end shoulder points are the end points

        shoulderWidth = self.right_beginning[calibrate][0] - self.left_beginning[calibrate][0]
        return eyeWidth/shoulderWidth

    # returns ratio between eye width and shoulder height
    def getEyeShoulderHeightRatio(self, calibrate = False):
        eyeWidth = self.right_eye[calibrate][0] - self.left_eye[calibrate][0]
        maxShoulder = max(max(self.right_end[calibrate][1],self.left_end[calibrate][1]),max(self.right_beginning[calibrate][1],self.left_beginning[calibrate][1]))
        shoulderHeight = maxShoulder - max(self.right_eye[calibrate][1], self.left_eye[calibrate][1])
        return np.abs(eyeWidth/shoulderHeight)

    @staticmethod
    def sigmoid(num):
        return 1/(1 + np.exp(-num))
