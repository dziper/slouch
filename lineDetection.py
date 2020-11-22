import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import datetime

def tryLine(x,y,low,high, plot = False, getLine = False):
    totalLength = len(x)
    x_slice = x[int(totalLength*low):int(totalLength*high)]
    y_slice = y[int(totalLength*low):int(totalLength*high)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_slice,y_slice)
    corrCoeff = r_value ** 2
    line = intercept + slope * x_slice
    # print("low {} high {} corr {}".format(low,high,corrCoeff))

    if plot:
        plt.plot(x_slice,y_slice)
        plt.plot(x_slice, line)

    if getLine:
        return corrCoeff, line, slope, intercept
    else:
        return corrCoeff

def findBestFit(x,y,low=0.35,high=0.65):
    corrStepThresh = -0.01
    corr = tryLine(x,y,low,high)
    prevScore = 0
    iters = 0
    max_iters = 20
    change = -0.2

    sizeWeight = 0.3

    for i in range(max_iters):
        corr = tryLine(x,y,low,high)
        score = (1-sizeWeight + sizeWeight*(high - low)) * corr
        # print("score {}".format(score))

        if score < prevScore + corrStepThresh:
            change = change * -1

        low += change

        change *= 0.7
        prevScore = score

        if low < 0 or low > high:
            # print("resetting low")
            low = 0

    low = low - change
    if low < 0 or low > high:
        low = 0
    # print()

    change = 0.2
    for i in range(max_iters):
        # print(high)
        corr = tryLine(x,y,low,high)
        score = (1-sizeWeight + sizeWeight*(high - low)) * corr
        # print("score {}".format(score))


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
