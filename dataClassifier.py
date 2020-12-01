import math

class DataClassifier:
    def __init__(self,frameData, calibratedData):
        if len(frameData) != 8 or len(calibratedData) != 8:
            return False
        self.right_eye = (frameData[0], calibratedData[0])
        self.left_eye = (frameData[1], calibratedData[1])
        self.right_beginning = (frameData[2], calibratedData[2])
        self.right_end = (frameData[3], calibratedData[3])
        self.right_slope = (frameData[4], calibratedData[4])
        self.left_beginning = (frameData[5], calibratedData[5])
        self.left_end = (frameData[6], calibratedData[6])
        self.left_slope = (frameData[7], calibratedData[7])

        return True

    #update data
    def newData(self, data, classify = False):
        if len(frameData) != 8 or len(calibratedData) != 8:
            return False

        self.right_eye[classify] = frameData[0]
        self.left_eye[classify] = frameData[1]
        self.right_beginning[classify] = frameData[2]
        self.right_end[classify] = frameData[3]
        self.right_slope[classify] = frameData[4]
        self.left_beginning[classify] = frameData[5]
        self.left_end[classify] = frameData[6]
        self.left_slope[classify] = frameData[7]

        return True

    def classify(self):
        weightedSum = 0

    #returns angle difference between calib and frame angle in degrees
    def getAngleDifference(self):
        calibSlopeDiff = self.right_slope[1] - self.left_slope[1]
        calibAngleBetween = np.arctan2(calibSlopeDiff,1) * 180 / math.pi

        slopeDiff = self.right_slope[0] - self.left_slope[0]
        angleBetween = np.arctan2(slopeDiff,1) * 180 / math.pi

        return calibAngleBetween - angleBetween

    # returns ratio beterrn eyes and shoulders
    def getEyeShoulderRatio(self, classify=False):
        eyeWidth = self.right_eye[classify][0] - self.left_eye[classify][0]

        # TODO: Check if left/right beginninin or end shoulder points are the end points

        shoulderWidth = self.right_beginning[classify][0] - self.left_beginning[classify][0]
        return eyeWidth/shoulderWidth
