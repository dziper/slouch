import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import datetime

length = 100
split1 = 20
split2 = 70
CORR_COEFF_THRESH = 0.95

# set up data
x = np.arange(length)
y = np.array([])

for i in range(split1):
    y = np.append(y, i*4)
for i in range(split2 - split1):
    y = np.append(y, y[split1 - 1] + i*1.2)
for i in range(length - split2):
    y = np.append(y, y[split2 - 1] + i*0.2)

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



low = 0
high = 1

corrStepThresh = -0.01
corr = tryLine(x,y,low,high)
prevScore = corr
iters = 0
max_iters = 10
low = 0
change = 0.1

for i in range(max_iters):
    corr = tryLine(x,y,low,high)
    score = (0.9 + 0.1*(high - low)) * corr
    print("score {}".format(score))

    low += change

    if score < prevScore + corrStepThresh:
        change = change * -1

    change *= 0.8
    prevScore = score

low = low - change
print()

change = 0.1
for i in range(max_iters):
    corr = tryLine(x,y,low,high)
    score = (0.8 + 0.1*(high - low)) * corr
    print("score {}".format(score))

    if score < prevScore + corrStepThresh:
        change = change * -1

    change *= 0.8
    high += change
    prevScore = score

high = high - change

plt.plot(x,y)
line = intercept + slope * x
plt.plot(x, line)
corr = tryLine(x,y,low,high, plot = True)
plt.show()
