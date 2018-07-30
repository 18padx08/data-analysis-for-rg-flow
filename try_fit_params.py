import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import os

basepath = "./scripts/data/phi4/compare_network_mc/"

def func(x, A,lamb, kappa):
    return A*np.exp(-(1-2*kappa) * x**2 - lamb * (x**2 -1)**2)

data = []

Aparam = 0
B = 0
for root,dirs,files in os.walk(basepath):
    for file in files:
        isFirst = True
        data = []
        theCsv = csv.reader(open(os.path.join(root,file)))
        for line in theCsv:
            if isFirst:
                i = 0
                for ele in line:
                    if ele != "":
                        data += [float(ele)]
                        #data += [-float(ele)]
                isFirst = False
            else:
                Aparam = float(line[0])
                B = float(line[1])
                Rescaling = 1
                break
        data = np.array(data)
        #data = data * Rescaling
        hist,bin_edges = np.histogram(data,bins=1000, density=True)
        print(np.all(np.diff(bin_edges)==1))
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.0
        #print(B, Aparam)
        (A,lamb,kappa), pcov = curve_fit(func,bin_centers,hist,p0=[np.max(hist), B, Aparam])
        plt.plot(bin_centers,hist)
        xs = np.linspace(-4,4,1000)
        #print(lamb,kappa)
        plt.plot(xs, func(xs,A,lamb,kappa), label="Amp="+str(int(A)) + " lam=" + str(round(lamb,2)) + " kappa="+str(round(kappa,2)))
        plt.plot(xs, func(xs, A, B, Aparam), label="B=" +str(round(B,2)) + " A="+str(round(Aparam,2)))
        plt.title(file)
        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$P(\phi)$")
        plt.legend()
        plt.savefig("./data/compare_network_mc/" + file + ".png")
        plt.clf()