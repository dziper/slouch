import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import datetime

# Tries a slice (defined by low and high values) of the data
# Returns the correlation coefficient (to evaluate how accurate the line is)
def tryLine(x,y,low,high, plot = False, getLine = False):
    totalLength = len(x)
    x_slice = x[int(totalLength*low):int(totalLength*high)]
    y_slice = y[int(totalLength*low):int(totalLength*high)]

    if x_slice is None or y_slice is None or x_slice.size == 0 or y_slice.size == 0:
        return None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_slice,y_slice)
    corrCoeff = r_value ** 2
    y_points = intercept + slope * x_slice
    line = np.stack((x_slice, y_points), axis=-1)
    # print("low {} high {} corr {}".format(low,high,corrCoeff))

    if plot:
        plt.plot(x_slice,y_slice)
        plt.plot(x_slice, line)
        plt.show()

    if getLine:
        return corrCoeff, line, slope, intercept
    else:
        return corrCoeff

# Iterate over many low and high values in order to maximize correlation
# and maximize the amount of data used
#
def findBestFit(x,y,low=0.35,high=0.65,plot = False):
    corrStepThresh = -0.01
    corr = tryLine(x,y,low,high)
    if corr == None:
        return None
    prevScore = 0
    iters = 0
    max_iters = 10
    change = -0.2

    sizeWeight = 0.2

    # Maximize correlation and data relative to `low`
    for i in range(max_iters):
        corr = tryLine(x,y,low,high)
        if corr == None:
            corr = 0
        score = (1-sizeWeight + sizeWeight*(high - low)) * corr
        # Calculate score of line

        if score < prevScore + corrStepThresh:
            change = change * -1

        low += change

        change *= 0.7
        prevScore = score

        if low < 0 or low > high:
            low = 0

    low = low - change
    if low < 0 or low > high:
        low = 0


    # Maximize correlation and data relative to `low`
    change = 0.2
    for i in range(max_iters):
        corr = tryLine(x,y,low,high)
        if corr == None:
            corr = 0
        score = (1-sizeWeight + sizeWeight*(high - low)) * corr
        # Calculate score of line

        if score < prevScore + corrStepThresh:
            change = change * -1
        high += change

        change *= 0.7
        prevScore = score
        if high < low or high > 1:
            high = 1

    high = high - change
    if high < low or high > 1:
        high = 1


    return tryLine(x,y,low,high,getLine = True,plot=plot)
