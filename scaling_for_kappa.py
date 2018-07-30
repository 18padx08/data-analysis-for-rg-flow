import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def func(x, kappa):
    return ((np.cos(x/22 * 2*3.141)* np.exp(-x/44) + 1)/4 +0.5) *kappa 

basepath= "./scripts/data/scalings/newbs"
allPoints = []

for root, dirs, files in os.walk(basepath):
    for file in sorted(files):
        print(file)
        tmpList = []
        theCsv = csv.reader(open(os.path.join(root,file)))
        for line in theCsv:
            for ele in line:
                if ele != "":
                    tmpList += [float(ele)]
        allPoints += [tmpList]
lambdas = [0,0.5,1,2,5,7,9,11,12,13,14,16,18,20,25,30,35,40]
i = 0
means = []
stds = []
for points in allPoints:
    means += [np.mean(points)]
    stds += [np.std(points)]
    #plt.errorbar(lambdas[i], np.mean(points),yerr=np.std(points),capsize=5,fmt="bo")
    i+=1
means = np.array(means)
stds = np.array(stds)

plt.fill_between(lambdas[:len(means)],means+stds,means-stds, interpolate=True, alpha=0.3, linestyle="-")
plt.plot(lambdas[:len(means)], means,"bo")
xs = np.linspace(0,50,1000)
#plt.plot(xs,func(xs,0.3))
plt.xlabel(r"$B(\lambda)$")
plt.ylabel(r"$A (\kappa)$")
plt.show()
