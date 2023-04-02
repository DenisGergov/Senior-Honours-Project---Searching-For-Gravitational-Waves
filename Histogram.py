"""
Histogram Code - Spectrogram
S1839787 Denis Gergov

"""
import gzip
import numpy as np
import math
import matplotlib.pyplot as plt
import pylab as plot

def readFile(fileName):
    file = gzip.open(fileName, "rt", encoding = "utf-8")
    array = np.array(file.read().splitlines())
    float_array = array.astype(np.float)
    float_array = float_array.reshape(257, 2545)
    file.close()
    return float_array

array = readFile("ligospec.dat.gz")
print(np.shape(array))

"""
plt.imshow(array, cmap = 'viridis', aspect = 'auto', interpolation = 'nearest')
plt.xlim(0, 2545)
plt.ylim(0, 257)
plt.xlabel('Time', labelpad = 12)
plt.ylabel('Frequency', labelpad = 4)
plt.colorbar()
plt.show()
"""

x = np.linspace(-10, 10 ,2545)
y = np.linspace(0, 2000, 257)
X, Y = np.meshgrid(x, y)
plt.pcolormesh(X, Y, array)
plt.colorbar()
plt.xlabel('Time', labelpad = 12)
plt.ylabel('Frequency', labelpad = 4)
plt.savefig('spec.png')
plt.show()

fs = 4096
deltat = 10
NFFT = fs/8
NOVL = NFFT*15/16
window = np.blackman(NFFT)
spec_cmap = 'viridis'
#spec_H1, freqs, bins, im = plt.specgram(array, NFFT=NFFT, Fs=fs, window=window, 
#                                        noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat,deltat])



def Whiten(spec):
    
    for i in range(257):
        sum = 0
        for j in range(2545):
            sum += (spec[i, j])
        sum = sum/float(2545)
        for j in range (2545):
            spec[i,j] = spec[i,j]/sum
    return spec

whitened = Whiten(array)

"""
plt.imshow(whitened, cmap = 'viridis', aspect = 'auto', interpolation = 'nearest')
plt.xlim(0, 2546)
plt.ylim(0,257)
plt.xlabel('Time', labelpad = 12)
plt.ylabel('Frequency', labelpad = 4)
plt.colorbar()
plt.show()
"""

x = np.linspace(-10, 10 ,2545)
y = np.linspace(0, 2000, 257)
X, Y = np.meshgrid(x, y)
plt.pcolormesh(X, Y, whitened)
plt.xlabel('Time', labelpad = 12)
plt.ylabel('Frequency', labelpad = 4)
plt.colorbar()
plt.savefig('spec_white.png')
plt.show()

"""
plt.xlim([-1, 1])
plt.ylim([0,300])
"""


maxElement = np.amax(whitened)
print('Max element from Numpy Array : ', maxElement)
whitened_flat = np.reshape(whitened, 654065)

print("THE SHAPE OF THE FLAT ARRAY ", np.shape(whitened))

std = np.std(whitened_flat)

print("THE MEAN AND STANDARD DEVIATION OF THE HISTOGRAM are ", str(np.mean(whitened_flat)), str(std))
print("The max value lies ", str((maxElement/std) - 1), "standard deviations away")
prob = math.exp(-maxElement)
print("The prob of value > the max vaulue is of that pixel is ", prob)
print("The prob of value > the max value anywhere on the graph is ", 654065*prob)

y = np.linspace(0, 30000, 100)
x = np.ones(100)
x = x*std

np.histogram(whitened_flat, bins=10, range=None, normed=None, weights=None, density=None)

plt.hist(whitened_flat, bins=100)
plt.xlabel("x = P/<P>")
plt.ylabel("N(x)")
plt.xlim([0, 10])
#plt.ylim([7999, 9000])
print("The maximum histogram value is", y.max())


model_x = np.linspace(0, 10, 1000)
model_y = np.ones(1000)
for i in range(len(model_x)):
    model_y[i] = 100000*math.exp((-1)*model_x[i])
plt.plot(model_x,model_y,'red',linewidth=2)

plt.title("Whitened Signal Power Distribution")
plt.savefig("Histogram.png")
plt.show()

#NOW WE BEGIN COMBINING PIXELs
def pixReduce(z, n, m):
    
    z2 = np.zeros((n//2 + 1, m//2 + 1 ))
    for i in range(n):
        for j in range(m):
            isub = i//2
            jsub = j//2
            z2[isub, jsub] += z[i, j]/4
    return z2

shape_w = list(np.shape(whitened))
red = pixReduce(whitened, shape_w[0], shape_w[1] )

maxReduced = np.amax(red)
print("THE MAX VALUE OF THE REDUCED RESOLUTION ARRAY IS ", maxReduced)

x = np.linspace(-10, 10 ,1273)
y = np.linspace(0, 2000, 129)
X, Y = np.meshgrid(x, y)
plt.pcolormesh(X, Y, red)
plt.xlabel('Time', labelpad = 12)
plt.ylabel('Frequency', labelpad = 4)
plt.colorbar()
plt.savefig('spec_white_reduced.png')
plt.show()

shape_red = list(np.shape(red))
flat = np.reshape(red, shape_red[0]*shape_red[1])
plt.hist(flat, bins=150)
plt.xlim([0, 10])

std = np.std(flat)

print("THE MEAN AND STANDARD DEVIATION OF THE  REDUCED HISTOGRAM are ", str(np.mean(flat)), str(std))
print("The max value lies ", str((maxReduced/std) - 1), "standard deviations away")
prob = (1 + 2*maxReduced)*math.exp((-2)*maxReduced)
print("The prob of value > the max vaulue is of that pixel is ", prob)
print("The prob of value > the max value anywhere on the graph is ", 1273*129*prob)

model_x = np.linspace(0, 10, 1000)
model_y = np.ones(1000)
for i in range(len(model_x)):
    model_y[i] = 14500*4*model_x[i]*math.exp((-2)*model_x[i])
plt.plot(model_x,model_y,'red',linewidth=2)

plt.savefig('histogram_reduced.png')
plt.show()
