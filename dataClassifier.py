import math

class DataClassifier:
    right_eye = None
    left_eye = None

    right_beginning = None
    right_end = None
    right_slope = None

    left_beginning = None
    left_end = None
    left_slope = None

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

    def classify(self):
        weightedSum = 0

    #returns angle difference between calib and frame angle in degrees
    def getAngleDifference(self):
        calibSlopeDiff = self.right_slope[1] - self.left_slope[1]
        calibAngleBetween = np.arctan2(calibSlopeDiff,1) * 180 / math.pi

        slopeDiff = self.right_slope[0] - self.left_slope[0]
        angleBetween = np.arctan2(slopeDiff,1) * 180 / math.pi

        return calibAngleBetween - angleBetween

    # returns
    def getEyeShoulderRatio(index):
