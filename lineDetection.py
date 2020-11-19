import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import datetime

length = 100
split1 = 20
split2 = 70
CORR_COEFF_THRESH = 0.95

x = np.arange(length)
y = np.array([])

for i in range(split1):
    y = np.append(y, i*4)
for i in range(split2 - split1):
    y = np.append(y, y[split1 - 1] + i*1.2)
for i in range(length - split2):
    y = np.append(y, y[split2 - 1] + i*0.5)

for i in range(length):
    y[i] = y[i] + random.random() * random.randint(-5,5)


print(datetime.datetime.now())
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print(datetime.datetime.now())
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


low = 0.25
high = 0.75

corrStepThresh = 0.01
corr = tryLine(x,y,low,high)
prevCorr = corr
iters = 0
max_iters = 10

while corr < CORR_COEFF_THRESH and iters < max_iters:

    corr = tryLine(x,y,low,high)

    if corr > prevCorr - corrStepThresh:


    prevCorr = corr
    iters += 1

tryLine(x,y,0.25,0.75)
