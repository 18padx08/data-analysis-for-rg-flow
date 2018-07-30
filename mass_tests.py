import numpy as np
import matplotlib.pyplot as plt
import csv
for kappa in ["0.400000","0.470000"]:
    for cs in ["2048"]:
        try:
            theCsv = csv.reader(open("./scripts/data/phi4/field_scaling/mass_test_cs="+str(cs)+"_kappa=" + kappa + ".csv"))
        except (Exception):
            continue
        rs = range(1,19)
        xs = []
        for line in theCsv:
            tmp = []
            for r in rs:
                tmp += [float(line[r])]
            xs += [tmp]

        xs = np.array(xs)
        xs = xs.T
        newXs = []
        for y in range(xs.shape[0]):
            newXs += [np.mean(xs[y,:])]

        newXs = np.array(newXs)
        newXs = np.log(newXs[newXs > 0.05])
        rs = np.array(range(1,len(newXs)+1))
        pol = np.polyfit(rs,newXs,deg=1)

        plt.plot(rs,newXs, "ro")
        linx = np.linspace(1,20, 1000)
        plt.plot(linx, linx * pol[0] + pol[1], "r-",label=str(round(pol[0], 2)) + "x +" + str(round(pol[1],2)))
        if(float(kappa) == 0.4):
            newkappa = 1.0/(1.0/(4*float(kappa)) + 3.0/2.0)
            print(newkappa)
            scalefactor = 1.0/np.sqrt(((1.0/float(kappa))-2)/((1.0/newkappa) - 2))
            print(scalefactor)
            plt.plot(linx, linx * scalefactor*pol[0] + pol[1] + np.log(2 * float(kappa)/newkappa), "g-", label=str(round(scalefactor*pol[0],2)) + "x " + str(round(pol[1] + np.log(2*float(kappa)/newkappa),2)))
        plt.plot(linx, -linx * np.sqrt(1.0/float(kappa) - 2), "b-", label=r"$-\sqrt{\frac{1}{" +kappa + r"} - 2}\cdot r$")
        plt.legend()
        plt.xlabel("r")
        plt.ylabel(r"$\log{\left<\phi(0)\phi(r)\right>}$")
plt.show()