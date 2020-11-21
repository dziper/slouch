import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import datetime

length = 100
split1 = 20
split2 = 80
CORR_COEFF_THRESH = 0.95

# set up data
x = np.arange(length)
y = np.array([])

for i in range(split1):
    y = np.append(y, i*10)
for i in range(split2 - split1):
    y = np.append(y, y[split1 - 1] + i*1.2)
for i in range(length - split2):
    y = np.append(y, y[split2 - 1] + i*-1)

# add noise
for i in range(length):
    y[i] = y[i] + random.random() * random.randint(-5,5)


slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
corrCoeff = r_value ** 2

print(slope)
print(intercept)
print(corrCoeff)

plt.plot(x,y)
line = intercept + slope * x
plt.plot(x, line)
plt.show()

def tryLine(x,y,low,high, plot = False):
    totalLength = len(x)
    x_slice = x[int(totalLength*low):int(totalLength*high)]
    y_slice = y[int(totalLength*low):int(totalLength*high)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_slice,y_slice)
    corrCoeff = r_value ** 2
    print("low {} high {} corr {}".format(low,high,corrCoeff))

    if plot:
        plt.plot(x_slice,y_slice)
        line = intercept + slope * x_slice
        plt.plot(x_slice, line)

    return corrCoeff



low = 0.35
high = 0.65

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
    print("score {}".format(score))



    if score < prevScore + corrStepThresh:
        change = change * -1

    low += change

    change *= 0.7
    prevScore = score

    if low < 0 or low > high:
        print("resetting low")
        low = 0

low = low - change
if low < 0 or low > high:
    low = 0
print()

change = 0.2
for i in range(max_iters):
    print(high)
    corr = tryLine(x,y,low,high)
    score = (1-sizeWeight + sizeWeight*(high - low)) * corr
    print("score {}".format(score))


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

plt.plot(x,y)
line = intercept + slope * x
plt.plot(x, line)
corr = tryLine(x,y,low,high, plot = True)
plt.show()
