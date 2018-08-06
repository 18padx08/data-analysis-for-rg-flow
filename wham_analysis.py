import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.optimize import curve_fit

size_based_data = {}
L = 10

def function(kappas, kc, nu, A):
    return ((kappas-kc)/kc) *L**nu 


def try_fit(kappas, data, kc0,nu0,mu0):
    pass

for folder in ["16_16","20_20","25_25","30_30","32_32"]:
    basepath = "./scripts/data/fss/fixed_kappa/" + folder

    size_based_data[folder] = {}
    size_based_data[folder]["phi2"] = np.load(os.path.join(basepath,"phi2.npy"))
    size_based_data[folder]["phi4"] = np.load(os.path.join(basepath,"phi4.npy"))
    size_based_data[folder]["phiAvg"] = np.load(os.path.join(basepath,"phiAvg.npy"))
    size_based_data[folder]["errors"] = np.load(os.path.join(basepath,"errors.npy"))
kappas = np.load(os.path.join(basepath,"kappas.npy"))

for size in size_based_data.keys():
    L= float(size[:2])
    plt.plot(kappas,size_based_data[size]["phiAvg"]*L**0.125, "-.", label=size)
    plt.fill_between(kappas,(size_based_data[size]["phiAvg"] - size_based_data[size]["errors"])*L**0.125,(size_based_data[size]["phiAvg"] + size_based_data[size]["errors"])*L**0.125, alpha=0.3)
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"$<|\phi|>L^{0.125}$")
    plt.legend()

crossings = []
vev = []
for i in range(1,len(size_based_data.keys())):
    firstKey = list(size_based_data.keys())[i-1]
    secondKey = list(size_based_data.keys())[i]
    #mask = abs(1-size_based_data[secondKey]["phiAvg"]/size_based_data[firstKey]["phiAvg"] ) <0.02
    mask =  abs(1- (size_based_data[secondKey]["phi2"]-1)**2/(size_based_data[firstKey]["phi2"] -1)**2) < 0.02
    nextToExpectedCrit =kappas[mask]
    secondMask =abs(nextToExpectedCrit - 0.1 ) < 0.3
    element = list(kappas).index(kappas[mask][secondMask][0])
    crossings += [kappas[mask][secondMask][0]]
    vev += [size_based_data[secondKey]["phiAvg"][element]]

print(np.mean(vev), np.std(vev), vev)
k = kappas[mask]
print(crossings)
y = list(kappas).index(crossings[1])

x = size_based_data[secondKey]["phiAvg"][y]
plt.errorbar(np.mean(crossings), 0.55,yerr=np.std(vev) ,xerr=np.std(crossings), fmt="o", capsize=3)
plt.show()

for size in size_based_data.keys():
    L= float(size[:2])
    mask = ((np.mean(crossings)-kappas)/np.mean(crossings) < 0) & ((np.mean(crossings)-kappas)/np.mean(crossings) > -0.08)
    fitKappas = kappas[mask]
    fitData = size_based_data[size]["phiAvg"]
    #pop,pcov = curve_fit(function, kappas, fitData, p0=[np.mean(crossings),1,0.125], maxfev=10000)
    xData = (np.mean(crossings)-kappas)/np.mean(crossings) *L**1
    yData = size_based_data[size]["phiAvg"]*L**0.125
    error = size_based_data[size]["errors"]*L**0.125
    plt.plot(xData,yData, "o", label=size)
    plt.fill_between(xData, yData-error,yData+error,alpha=np.mean(crossings))
    plt.xlabel(r"$\frac{" + str(round(np.mean(crossings),2)) + r"-\kappa}{"+str(round(np.mean(crossings),2))+r"}L$")
    plt.ylabel(r"$<|\phi|>L^{0.125}$")
    plt.legend()
plt.show()