import numpy as np
import matplotlib.pyplot as plt
import csv
import os

basePath = "./scripts/data/phi4/convergence_test/"

for dirpath,dirnames,filenames in os.walk(basePath):
    for file in filenames:
        kappas = []
        lambdas = []
        nncorr = []
        mccorr = []
        theCsv = csv.reader(open(os.path.join(dirpath,file)))
        for line in theCsv:
            kappas += [float(line[0])]
            lambdas += [float(line[1])]
            nncorr += [float(line[2])]
            mccorr += [float(line[3])]
        theMean = np.mean(nncorr[-500:])
        plt.plot(nncorr,label="NN corr")
        theMeanArray = [theMean] * len(nncorr)
        theMeanArray = np.array(theMeanArray)
        xs = np.linspace(0,len(nncorr),1500)
        plt.plot(kappas, label="kappa")
        if np.mean(lambdas) <= 1:
            plt.plot(lambdas, "r-", label="lambda")
        plt.plot([theMean] * len(nncorr), "g-", label="NN mean")
        plt.fill_between(xs,[theMean] * len(nncorr) + np.std(nncorr), [theMean] * len(nncorr) - np.std(nncorr),color="green",alpha=0.3, label=r"Confidence Interval ($\pm$ " + str(round(np.std(nncorr),2))+r")", zorder=1000)
        plt.plot(mccorr, label="MC corr")
        plt.xlabel("MC time")
        plt.ylabel("Value")
        plt.title(file)
        plt.legend()
        plt.savefig("./data/conv_test/" + file+".png")
        plt.clf()
        #plt.show()