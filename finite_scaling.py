import numpy as np
import matplotlib.pyplot as plt 
import csv
import os
import re
from scipy.optimize import curve_fit

basepath = "./scripts/data/phi4_2d/fss_l=1/"
def log(b, a=np.exp(1)):
    return np.log(b)/np.log(a)
L = 20
def function(x,nu, bc,A):
    return (A*np.abs(x-bc))*L**(-nu)
B4sx = []
B4sy = []
for root,dirs,files in os.walk(basepath):
    print(files)
    for file in files:
        if ".csv" not in file:
            continue
        xs=[]
        ys=[]
        theCsv = csv.reader(open(os.path.join(root,file)))
        pattern = re.compile(r".*_(?P<chainsize>\d+)\.csv")
        match = pattern.match(file)
        L = float(match['chainsize'])
        print(L)
        for line in theCsv:
            xs += [float(line[1])]
            ys += [1-1.0/3.0*float(line[2])]
        xs = np.array(xs)
        ys = np.array(ys)
        #mask = (xs > 0.249) & (xs < 0.253)
        #newxs = xs[mask]
        #newys = ys[mask]
        popt,cov = curve_fit(function, xs,ys, p0=[1,0.25,2.0/3.0])
        print(popt)
        newx = np.linspace(0.24, 0.26, 1000)
        #plt.plot(newx, function(newx,popt[0],popt[1], popt[2]), label="chainsize=" + match['chainsize'])
        plt.plot(xs,ys, "o", label="chainsize="+match['chainsize'])
        plt.xlabel(r"$\kappa$")
        plt.ylabel(r"$B_4$")
        plt.legend()
#plt.savefig("./data/phi4/2d_study/mc_perf/mc_perf_lambda=0.png")

plt.show()

