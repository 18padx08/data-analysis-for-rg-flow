import numpy as np
import matplotlib.pyplot as plt
import csv
import os
basepath = "./scripts/data/phi4/phi2d_crit/phi2d/phi2d_crit/"
criticalPoints = []
for root, dirs, files in os.walk(basepath):
    for file in files:
        if not "csv" in file:
            continue
        theCsv = csv.reader(open(os.path.join(root,file)))
        tmpx = []
        tmpy = []
        lamb = 0
        for line in theCsv:
            lamb = float(line[0])
            tmpx += [float(line[1])]
            tmpy += [float(line[2])]
        #plt.plot(tmpx, tmpy, "ro", label = str(lamb))
        #plt.xlabel("kappa")
        #plt.ylabel("vev")
        #plt.show()
        tmpy = np.array(tmpy)
        tmpx = np.array(tmpx)
        basevalue = tmpy[0]
        for i in range(0,len(tmpx)-1):
            if tmpy[i+1]>5*basevalue:
                critPoint = i
                break
        criticalPoints += [(lamb,tmpx[critPoint] )]
        print(lamb)
print(criticalPoints)
xs = []
ys = []
for p in sorted(criticalPoints):
    xs += [p[0]]
    ys += [p[1]]
plt.plot(xs,ys, "r-")
plt.plot(xs,ys,"b*")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\kappa$")
plt.legend()
plt.show()
