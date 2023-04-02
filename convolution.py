"""
GW CONVOLUTION - Performs Convolution of LIGO data with Waveform template.

Denis Gergov S1839787
"""

import gzip
import numpy as np
import shutil
import math
import matplotlib.pyplot as plt
from scipy import signal

"""
with gzip.open('denis1.dat.gz', 'rb') as f_in:
    with open('STRAIN_H1.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        
with gzip.open('denis2.dat.gz', 'rb') as f_in:
    with open('TEMPLATE.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        
"""

def READ(filename):
    f = open(filename, 'r')
    file = f.readlines()
    DATA = np.zeros((len(file), 2))
    for l in range(len(file)):
        split = file[l].split()
        DATA[l, 0] = float(split[0])
        DATA[l, 1] = float(split[1])
    f.close()
    return DATA

def rms(y):
    ms = 0
    for i in range(len(y)):
        ms += y[i]**2
    ms = ms/len(y)
    rms = math.sqrt(ms)
    return rms

def signal_noise(y):
    variance = rms(y)
    sigMax = np.max(y)
    sigNoise = sigMax/variance
    return sigNoise


strain_H1 = READ('STRAIN_H1.txt')
NRtemp = READ('TEMPLATE.txt')

x1, y1 = strain_H1[:, 0], strain_H1[:, 1]
plt.plot(x1, y1, 'r')
#plt.xlim(-0.2, 0.2)
plt.ylim(-4,4)
#plt.ylim(-3,3)
#plt.ylim((-10, 10))
plt.xlabel('Time leading up to event (s)')
plt.ylabel('Strain (whitened)')
plt.title('GW event signal vs Time')
plt.savefig('11-close.png')
plt.show()



t_sample = x1[1]-x1[0]
print("SAMPLING RATE ", t_sample)
print(int(15/t_sample))


x2, y2 = NRtemp[:, 0], NRtemp[:, 1]

plt.plot(x2, y2, 'g')

#plt.xlim(-0.15, 0.025)
#plt.ylim((-10, 10))
plt.xlabel('Time leading up to event (s)')
plt.ylabel('Strain (whitened)')
plt.title('GW NR template vs Time')
plt.xlim(-0.65,0.1)
plt.savefig('2.png')
plt.show()

#CODE THAT DEALS WITH STRETCHING THE TEMPLATE
alpha = 1
N = np.shape(y2)[0]
X_new = np.linspace(x2[0], x2[N-1], int(alpha*N))
Y_new = np.interp(X_new, x2, y2) 
X_new = X_new*alpha

print('THE STRETCHED SAMPLING IS', X_new[1]-X_new[0])

plt.plot(X_new, Y_new, 'g')
plt.xlim(-0.65,0.1)
plt.xlabel('Time leading up to event (s)')
plt.ylabel('Strain (whitened)')
plt.title('NR template (stretched)')
plt.savefig('stretch')
plt.show()



#decides the trim in indices, of the conolution by 2 seconds at either side to eliminate the abnormal noise spikes
trim = int(2/t_sample)
print("NUMBER OF INDEX OFFSET ", 0.14/t_sample)
#calculate the convolution, then trim either side
#also note the template is flipped as matched filtering does not require a flip however scipy signal automatically flips the template, meaning we need to account for that
#convolution = signal.convolve(y1, np.flipud(y2), mode='full')
#convolution = convolution[(trim):(np.shape(convolution)[0] + 1 - trim)]

convolution = np.convolve(y1, np.flipud(Y_new) )
convolution = convolution[(trim):(np.shape(convolution)[0] + 1 - trim)]

#calculates the number of data points in the convolution
length = np.shape(convolution)[0]
length2 = length/2
print(length)
print(length/2)
#sets the time axis for plotting the convolution
xConv = np.linspace(-t_sample*((length)/2 ), t_sample*((length)/2 ), length)

print('THE CONVOLUTION SAMPLING IS', xConv[1]-xConv[0])

plt.plot(xConv, convolution, 'magenta')
plt.xlim(-15, 15)
#plt.xlim(-15.1, 15.1)
plt.xlabel('Time Offset (s)')
plt.ylabel('Convolution')
plt.title('(with NR template)')
plt.suptitle('Convolution of GW signal vs time offset')
plt.savefig('3.png')
plt.show() 

#calculates the signal to noise of the convolution peak
sig_noise = signal_noise(convolution)
print("The Signal/Noise of the convolution is ", sig_noise)


#plots a histogram of the convolution data points to confirm that the noise is drawn from a gaussian (central limit theorem)
#plots a gaussian distribution over it
rms_conv = rms(convolution)
plt.hist(convolution, bins = 500)
x = np.linspace(-100,100, 1000)
y = np.zeros((1000))
for i in range(np.shape(x)[0]):
    y[i] = 120000*(1/(rms_conv*math.sqrt(2*math.pi)))*math.exp(-((x[i]/rms_conv)**2)/(2))
plt.plot(x, y, 'r')
plt.xlabel('Noise Value')
plt.ylabel('Noise Distribution')
plt.title('Noise Distribution Histogram of Convolution')
plt.xlim(-100, 100)
plt.savefig('4.png')
plt.show()

#calculating probabilities
prob = (1/(sig_noise*math.sqrt(2*math.pi)))*math.exp(-(sig_noise**2)/(2))
print("The prob of a signal to noise at that time is ", prob)
prob_pix = prob* 2*int(15/t_sample)
print("The prob of a signal to noise over the whole experiment is ", prob_pix)


#same as above but calculates convoluting the template with itself
NRconv = signal.convolve(y2, y2[::-1], mode='full')
xConv = np.linspace(-(t_sample*np.shape(NRconv)[0])/2, (t_sample*np.shape(NRconv)[0])/2, np.shape(NRconv)[0])
plt.plot(xConv, NRconv, 'hotpink')
plt.xlim(-0.2, 0.2)
#plt.ylim(-175, 175)
plt.title('Convolution of NR template with itself')
plt.xlabel('Time Offset (s)')
plt.ylabel('Convolution')
plt.savefig('self.png')
plt.show()

sig_noise = signal_noise(NRconv)
print("The Signal/Noise of the NR Template convolution is ",sig_noise)



#code that plots the stretched s/n
trim = int(5/t_sample)
alpha = np.arange(0.2, 5.2, 0.01)
#alpha = np.log()
#alpha = np.array([0.25, 0.33333333, 0.5, 0.6666666, 0.75, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 5.0])
signal_noise_plot = np.zeros((np.shape(alpha)[0]))
N = np.shape(y2)[0]

for i in range(np.shape(alpha)[0]):
    X_new = np.linspace(x2[0], x2[N-1], int(N/alpha[i]))
    Y_new = np.interp(X_new, x2, y2) 
    X_new = X_new/alpha[i]
    
    convolute = np.convolve(y1, np.flipud(Y_new) )
    convolute = convolute[(trim):(np.shape(convolute)[0] + 1 - trim)]
    sig_noise = signal_noise(convolute)
    signal_noise_plot[i] = sig_noise
    
plt.plot(alpha, signal_noise_plot, 'limegreen')
plt.xlabel('α')
plt.ylabel('S/N')
plt.title('S/N as it varies by scaling NR template by α')

x = np.linspace(0,6, 1000)
y = np.zeros((1000))
for i in range(1000):
    y[i] = 5*(1/(0.1*math.sqrt(2*math.pi)))*math.exp(-(((x[i]-1)/0.1)**2)/(2))
plt.plot(x, y, 'r')

plt.savefig('5.png')
plt.show()



#Importing own template and convoluting

temp = READ('temp.txt')
#temp = READ('temp.txt')
temp_x, temp_y = temp[:, 0], temp[:, 1]
plt.plot(temp_x, temp_y, 'mediumslateblue')
plt.xlim(-0.1, 0.02)
#plt.ylim(-4,4)
#plt.ylim(-3,3)
#plt.ylim((-10, 10))
plt.xlabel('Time leading up to event (s)')
#plt.plot(x2, y2, 'b')          
plt.ylabel('Strain')
plt.title('Scaled GW template vs Time')
plt.savefig('6.png')
plt.show()


t_sample = temp_x[1]-temp_x[0]
trim = int(2/t_sample)
TEMPconv = signal.convolve(y1, np.flipud(temp_y), mode='full')
TEMPconv = TEMPconv[(trim):(np.shape(TEMPconv)[0] + 1 - trim)]
xConv = np.linspace(-t_sample*(np.shape(TEMPconv)[0])/2, (t_sample*np.shape(TEMPconv)[0])/2, np.shape(TEMPconv)[0])
plt.plot(xConv, TEMPconv, 'turquoise')
#plt.xlim(-5, 5)
#plt.ylim(-175, 175)
plt.xlabel('Time Offset (s)')
plt.ylabel('Convolution')
plt.title('(with generated template)')
plt.suptitle('Convolution of GW signal v time offset')
plt.savefig('7.png')
plt.show() 

sig_noiseTEMP = signal_noise(TEMPconv)
print("THE SIGNAL TO NOISE FROM OWN TEMPLATE IS", sig_noiseTEMP)

rms_temp = rms(TEMPconv)
plt.hist(TEMPconv, bins = 500)
x = np.linspace(-2e-19,2e-19, 1000)
y = np.zeros((1000))
for i in range(np.shape(x)[0]):
    y[i] = 9.5e-17*(1/(rms_temp*math.sqrt(2*math.pi)))*math.exp(-((x[i]/rms_temp)**2)/(2))
plt.plot(x, y, 'r')
plt.xlabel('Noise Value (own template)')
plt.ylabel('Noise Distribution')
plt.title('Noise Distribution (Convolution with Template)')
plt.savefig('8.png')
plt.show()


TEMPconv = signal.convolve(temp_y, np.flipud(temp_y), mode='full')
xConv = np.linspace(-t_sample*(np.shape(TEMPconv)[0])/2, (t_sample*np.shape(TEMPconv)[0])/2, np.shape(TEMPconv)[0])
plt.plot(xConv, TEMPconv, 'blueviolet')
plt.xlim(-0.2, 0.2)
#plt.ylim(-175, 175)
plt.xlabel('Time Offset (s)')
plt.ylabel('Convolution')
plt.suptitle('Convolution of template with itself')
plt.savefig('13.png')
plt.show() 

sig_noiseTEMP = signal_noise(TEMPconv)
print("THE SIGNAL TO NOISE FROM OWN TEMPLATE CONVOLUTED WITH ITSELF IS", sig_noiseTEMP)



