import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize
import csv
import re
from math import log10, floor
regex = r".*kappa=(?P<kappa>\d+\.\d{6}).*lambda=(?P<lambda>\d+.\d{6}).*"
pattern = re.compile(regex)
basepath = "./scripts/data/phi4/compare_dist/"
def gauss(x, A, mean, var,lamb):
    return A*np.exp(-(1-2*var)*(x)**2 - lamb*(x**2 -1)**2)
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)
for root, dirs, files in os.walk(basepath):
    for file in files:
        mc = []
        nn = []
        theCsv = csv.reader(open(os.path.join(root,file)))
        first = True
        for line in theCsv:
            if first:
                first =False
                for ele in line:
                    if ele != "":
                        mc += [float(ele)]
            else:
                for ele in line:
                    if ele!="":
                        nn += [float(ele)]
        plt.subplot(2,2,1)
        mcHist, mcBins = np.histogram(mc, bins=1000)
        mcHist = mcHist / mcHist.sum()
        mcCenters = (mcBins[1:]+mcBins[:-1])/2.0
        nnHist, nnBins = np.histogram(nn, bins=1000)
        nnHist = nnHist / nnHist.sum()
        nnCenters = (nnBins[1:]+nnBins[:-1])/2.0
        match = pattern.match(file)
        if match is None:
            raise Exception("no lambda or kappa found")
        explamb = float(match.group('lambda'))
        expkap = float(match.group('kappa'))
        paramsMc, covMC = scipy.optimize.curve_fit(gauss, mcCenters,mcHist, p0=[3e-3,0,3e-1,explamb])
        paramsNN, covNN = scipy.optimize.curve_fit(gauss, nnCenters,nnHist, p0=[3e-3,0,3e-1,explamb])
        x = np.linspace(np.min(mcCenters), np.max(mcCenters), 1000)
        print(paramsMc)
        print(paramsNN)
        
        plt.plot(mcCenters,mcHist, label="mc")
        plt.plot(x, gauss(x, paramsMc[0],paramsMc[1],paramsMc[2], paramsMc[3]), "r:", label="MC-fit")
        plt.plot(x, gauss(x, paramsNN[0],paramsNN[1],paramsNN[2], paramsNN[3]),"b-.", label="NN-fit")
        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$P(\phi)$")
        plt.legend()

        plt.subplot(2,2,2)
        #plt.text(0,0.0, "mean="+str(round_sig(paramsMc[1],3)))
        plt.text(0,0.3, r"$\kappa="+str(round_sig(paramsMc[2],3))+r"$")
        plt.text(0,0.6,r"$\lambda="+str(round_sig(paramsMc[3],3)) +r"$")
        plt.ylim(-1,2)
        ax = plt.gca()
        ax.axis("off")
        plt.subplot(2,2,3)
        
        plt.plot(nnCenters, nnHist, label="nn")
        plt.plot(x, gauss(x, paramsMc[0],paramsMc[1],paramsMc[2], paramsMc[3]), "r:", label="MC-fit")
        plt.plot(x, gauss(x, paramsNN[0],paramsNN[1],paramsNN[2], paramsNN[3]),"b-.", label="NN-fit")
        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$P(\phi)$")
        plt.legend()
        
        plt.subplot(2,2,4)
        #plt.text(0,0.0, "mean="+str(round_sig(paramsNN[1])))
        plt.text(0,0.3, r"$\kappa="+str(round_sig(paramsNN[2])) + r"$")
        plt.text(0,0.6,r"$\lambda="+str(round_sig(paramsNN[3])) + r"$")
        plt.ylim(-1,2)
        ax = plt.gca()
        ax.axis("off")
        #plt.show()
        plt.savefig("./data/phi4/compare_dists/" + file + ".png")
        plt.clf()


