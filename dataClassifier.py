right_eye = None
left_eye = None

right_beginning = None
right_end = None
right_slope = None

left_beginning = None
left_end = None
left_slope = None


def readData(frameData, calibratedData):
    if len(frameData) != 8 or len(calibratedData) != 8:
        return False
    right_eye = (frameData[0], calibratedData[0])
    left_eye = (frameData[1], calibratedData[1])
    right_beginning = (frameData[2], calibratedData[2])
    right_end = (frameData[3], calibratedData[3])
    right_slope = (frameData[4], calibratedData[4])
    left_beginning = (frameData[5], calibratedData[5])
    left_end = (frameData[6], calibratedData[6])
    left_slope = (frameData[7], calibratedData[7])

def classify():
    weightedSum = 0

def getAngleDifference():
    calibSlopeDiff = right_slope[1] - left_slope[1]
    calibAngleBetween = np.arctan2(calibSlopeDiff,1) * 180 / math.pi

    slopeDiff = right_slope[0] - left_slope[0]
    angleBetween = np.arctan2(slopeDiff,1) * 180 / math.pi

    return calibAngleBetween - angleBetween
