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
    y = np.append(y, y[split2 - 1] + i*0.5)

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

def tryLine(x,y,low,high):
    totalLength = len(x)
    x_slice = x[int(totalLength*low):int(totalLength*high)]
    y_slice = y[int(totalLength*low):int(totalLength*high)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_slice,y_slice)
    corrCoeff = r_value ** 2
    print("low {} high {} corr {}".format(low,high,corrCoeff))
    return corrCoeff


low = 0
high = 1

corrStepThresh = 0.01
corr = tryLine(x,y,low,high)
prevCorr = corr
iters = 0
max_iters = 10
low = 0.5

while corr < CORR_COEFF_THRESH and iters < max_iters:
    if corr > prevCorr - corrStepThresh:
        low = low/2
    else:
        low = low+low/2

    corr = tryLine(x,y,low,high)

    prevCorr = corr
    iters += 1

CORR_COEFF_THRESH = 0.98
while corr < CORR_COEFF_THRESH and iters < max_iters:
    corr = tryLine(x,y,low,high)

    if corr > prevCorr - corrStepThresh:
        high = high - high/2
    else:
        high = high/2
    if high == low:
        print("bad")
    prevCorr = corr
    iters += 1
