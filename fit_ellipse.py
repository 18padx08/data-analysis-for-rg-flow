import numpy as np
from numpy.linalg import eig, inv
import csv
import matplotlib.pyplot as plt

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

R = np.arange(0,2*np.pi, 0.01)

creader = csv.reader(open("scripts/data/response_error_bs=10_cs=1024pm.csv"), delimiter=",")
errorNN = []
errorMC =[]
errorHidden = []
for line in creader:
    errorNN += [float(line[0])]
    errorMC += [float(line[1])]
    errorHidden += [float(line[2])]
errorNN = np.array(errorNN)
errorMC = np.array(errorMC)
errorHidden = np.array(errorHidden)

plt.plot(errorNN, errorMC,"ro")
plt.xlabel("NN error")
plt.ylabel("MC error")
plt.title("Visible correlation length of NNN")
plt.legend()

a = fitEllipse(errorNN,errorMC)
center = ellipse_center(a)
#phi = ellipse_angle_of_rotation(a)
phi = ellipse_angle_of_rotation2(a)
axes = ellipse_axis_length(a)
axes[1] = 0.08
print(center)
print(phi)
print(axes)

a, b = axes
xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
plt.plot(xx,yy)
plt.show()

plt.plot(errorHidden, errorMC,"ro")
plt.xlabel("NN error")
plt.ylabel("MC error")
plt.title("Hidden correlation length of NN")
plt.legend()

a = fitEllipse(errorHidden,errorMC)
center = ellipse_center(a)
#phi = ellipse_angle_of_rotation(a)
phi = ellipse_angle_of_rotation2(a)
axes = ellipse_axis_length(a)
axes[1] = 0.08
print(center)
print(phi)
print(axes)

a, b = axes
xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
plt.plot(xx,yy)
plt.show()