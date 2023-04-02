"""
GW TEMPLATE 
Code that creates a custom GW Binary Black Hole Merger template
Based on Analytical Newtonian Approximations of Ringdown
Plus solving snalytically the energy loss rate of GWs from GR.
Denis Gergov S1839787

"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fft import fft, fftfreq


# constants:
clight = 2.99792458e8                # m/s
G = 6.67259e-11                      # m^3/kg/s^2 
MSol = 1.989e30                      # kg

#masses to determine ratio of M1 and M2 (used GW150914 mass ratio as placeholder)
#M_1 = 36
#M_2 = 29

M_1 = 36
M_2 = 29

#chirp mass
bigM = (((M_1*M_2)**3)/(M_1 + M_2))**(1/5)

#dimensionless ringdown frequency
omegaR_dim = (((M_1*M_2)**(3/5))/((M_1 + M_2)**(6/5)))

#dimensionless ringdown time
tR_dim = ((-5)/(4**(8/3)))*omegaR_dim**((-8)/3)

#tau_R
tau = 4.2*(1/omegaR_dim)

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True

#inspiral before ringdown (dimensionless)
def f(x, phi0):
   return (4**(1/3))*(5**(1/4))*((-x)**(-1/4))*np.cos(4*(5**(-5/8))*((-x)**(5/8)) + phi0)

def gradient_f(x, phi0):
    return (4**(1/3))*(5**(1/4))*( (((-x)**((-5)/4))/4)*np.cos(4*(5**((-5)/8))*((-x)**(5/8)) + phi0) + ((((-x)**((-5)/8))*(5**(3/8)))/2)*np.sin(4*(5**((-5)/8))*((-x)**(5/8)) + phi0) )
    #return (5**(1/4) *np.cos((4*(-x)**(5/8))/5**(5/8) + phi0)/(2*2**(1/3)*(-x)**(5/4)) + (5**(5/8) *np.sin((4*(-x)**(5/8))/5**(5/8) + phi0)/(2**(1/3) *(-x)**(5/8))))

#x-axis (time) in dimensionless units
#x = np.linspace(-1002, 1002, 1232 )
x = np.linspace(-1002, 1002, 1137)

#arbitrary initial phase constant
phi0 = 0.4

#strain at matching point calculated using function before ringdown
h_Ri = f(tR_dim, phi0)

#derivative of strain before ringdown, evaluated at matching point
dh_Ri = gradient_f(tR_dim, phi0)

#initial phase for the strain function AFTER ringdown
phi = math.atan( (-1/omegaR_dim)*(dh_Ri/h_Ri + 1/tau) )

#constant A for the strain function AFTER ringdown
A = h_Ri/math.cos(phi)


print("The ringdown time is ", tR_dim)
print("The value of the strain at the matching point is ", h_Ri)
print("The derivative at the matching point is ", dh_Ri)
print("The constant A is ", A)
print("The phase angle is ", phi)
print("THE DIMENSIONS OF X are ", np.shape(x))

#Function of strain AFTER ringdown
def g(t):
   return A*(np.exp(-((t - tR_dim)/tau)))*np.cos(omegaR_dim*(t - tR_dim) + phi)

#gradient of strain after ringdown
def gradient_g(t):
    return ((-A)/tau)*(np.exp(-((t - tR_dim)/tau)))*np.cos(omegaR_dim*(t - tR_dim) + phi) -A*omegaR_dim*(np.exp((-1)*((t - tR_dim)/tau)))*np.sin(omegaR_dim*(t - tR_dim) + phi)


#test to check the gradient just before and just after the matching point
diff = (tR_dim)/100
print("TEST TEST TEST TEST TEST TEST TEST TEST")
print("BEFORE THE MATCH, the gradient is ", gradient_f((tR_dim - abs(diff)), phi0) )
print("AFTER THE MATCH, the gradient is ", gradient_g(tR_dim + abs(diff)) ) 

print("yplt1 gradient is ", gradient_f(tR_dim, phi0))
print("yplt2 gradient is ", gradient_g(tR_dim ) )


#array of function before ringdown initially zeros
yplt1 = np.zeros(int(np.shape(x)[0]))

#array of function after ringdown (also zeros)
yplt2 = np.zeros(int(np.shape(x)[0]))

#fills in array with function before ringdown
for i in range(len(x)):
    if x[i] <= tR_dim:
        yplt1[i] = f(x[i], phi0)

#fills in array with the function after ringdown
for j in range(len(x)):
    if x[j] > tR_dim:
        yplt2[j] = g(x[j])

#adds the two functions together to produce fina form of chirp, plots them
yplt = yplt1 + yplt2
#plt.plot(x, yplt1, 'b')
#plt.plot(x, yplt2, 'g')  
plt.plot(x, yplt, 'r') 
plt.title('(before and after ringdown)')
plt.suptitle('Dimensionless GW strain template')
plt.xlabel('Time (dimensionless units)')
plt.ylabel('Strain (dimensionless units)')
plt.xlim(-200, 60)
plt.show()

#this is for when I will scale the template into normal units, still not finished
distance = 1.265*(10**(25))
xplt_scaled = x*((G*bigM*MSol)/clight**3)
sc = (((G*bigM*MSol)/clight**3)/((distance)/clight))
print("The value of the y axis scale factor is ", sc)
yplt_scaled = sc*yplt

plt.plot(xplt_scaled, yplt_scaled, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('GW Template (Scaled to regular units)')
plt.xlim(-0.1, 0.1)
plt.show()

"""
#whiten the template?
N = np.shape(xplt_scaled)[0]
T = 0.000244140625

yf = fft(yplt_scaled)
xf = fftfreq(N, T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
#plt.xlim(0,100)
plt.show()

rang = np.shape(yf)[0]
sum = 0
for i in range (rang):
    sum += (abs(yf[i]))**2
print("SUM", sum)
mean = sum/rang
rms = math.sqrt(mean)
yf = yf/rms

yinv = np.fft.ifft(yf)

plt.plot(xplt_scaled, yinv, 'y')
#plt.xlim(-0.03, 0.03)
plt.xlabel('Time (s)')
plt.ylabel('Strain (whitened)')
plt.title('Whitened Strain vs Time')
plt.show()
"""

interval = xplt_scaled[1] - xplt_scaled[0]
Range = xplt_scaled[np.shape(xplt_scaled)[0]-1] - xplt_scaled[0]
print("Sampling interval is ", interval)
print(Range)
num = int(Range/0.000244140625)
print(num)



with open('test.txt', 'w') as f:
    for item in range(len(yplt_scaled)):
        f.write(str(xplt_scaled[item]) + ' ' + str(yplt_scaled[item]) + '\n' )
        #f.write("%s\n" % item)


      
