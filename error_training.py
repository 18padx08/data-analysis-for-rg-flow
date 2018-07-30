import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from numpy import polyfit

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

errorNN = [0.157181,0.08305,0.0895,0.0172875,0.24785,0.1486,0.120616,0.238188,0.208769,0.527744,0.339019,0.261978,0.0093625,0.0158562,0.0271969,0.0281469,0.107934,0.0212156,0.1314,0.197713]
errorMC = [0.078379,0.00636041,0.044156,0.02906,0.0151564,0.166119,0.028731,0.052031,0.0137189,0.101776,0.0363904,0.0763466,0.0358863,0.0256159,0.0558885,0.0363904,0.0174345,0.0969745,0.0324834,0.0656335]
poly = polyfit(errorNN, errorMC,1)
print(poly)
p = np.poly1d(poly)
xs = np.linspace(0, np.max(errorNN),1000)
plt.plot(xs, p(xs))
plt.plot(errorNN, errorMC,"ro")
plt.xlabel("NN error")
plt.ylabel("MC error")
plt.legend()
plt.show()

creader = csv.reader(open("scripts/data/error_scatter_bs=10_cs=1024pm.csv"), delimiter=",")
errorNN = []
errorMC =[]
for line in creader:
    errorNN += [float(line[0])]
    errorMC += [float(line[1])]
errorNN = np.array(errorNN)
errorMC = np.array(errorMC)
#linear regression
poly = polyfit(errorNN, errorMC, 1)
print(poly)
p = np.poly1d(poly)
xs = np.linspace(0, np.max(errorNN),1000)
plt.plot(xs, p(xs))
plt.plot(errorNN, errorMC,"ro")
plt.xlabel("NN error")
plt.ylabel("MC error")
plt.legend()
plt.show()

creader = csv.reader(open("scripts/data/response_error_bs=10_cs=1024pm.csv"), delimiter=",")
errorNN2 = []
errorMC2 =[]
for line in creader:
    errorNN2 += [float(line[0])]
    errorMC2 += [float(line[1])]
errorNN2 = np.array(errorNN2)
errorMC2 = np.array(errorMC2)
#linear regression
poly = polyfit(errorNN2*2, errorMC2, 1)
print(poly)
p = np.poly1d(poly)
xs = np.linspace(0, np.max(errorNN2),1000)
plt.plot(xs, p(xs))
plt.plot(errorNN2, errorMC2,"ro")
plt.xlabel("NN error")
plt.ylabel("MC error")
plt.legend()
plt.show()

plt.plot(errorNN, errorNN2*2, "ro")
plt.show()
plt.plot(errorMC,errorMC2, "ro")
plt.show()
creader = csv.reader(open("scripts/data/error_gauss_0.csv"), delimiter=",")
ys= []
for line in creader:
    ys += [float(line[0])]

hist, bin_edges = np.histogram(ys, bins=6)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
p0 = [1,-0,1]
coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
x = np.linspace(bin_centres[0], bin_centres[-1], 1000)
hist_fit = gauss(x, *coeff)
plt.plot(bin_centres, hist, label='Test data')
plt.plot(x, hist_fit, label='Fitted data')
print(np.abs(coeff[1]))
print(coeff[2])
plt.xlabel("A")
plt.ylabel("#events")
plt.legend()
plt.show()