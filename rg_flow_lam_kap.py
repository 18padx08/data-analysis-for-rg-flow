import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import csv

basepath = "./scripts/data/phi4/phi4_2d_rg_flow_c/"
basename = "rg_flow_comb_kappa="
basenameLam = "rg_flow_comb_lambda="
middlepart = "_bs=30_cs=1024_withLam="
middlepartLam =  "_bs=30_cs=1024_withKappa="
ending = ".csv"
fig, ax = plt.subplots()

for kappa in ["0.100000","0.150000","0.200000","0.250000","0.300000","0.350000","0.400000","0.450000","0.500000","0.550000","0.600000","0.700000","0.800000","0.900000","1.000000","1.100000","1.200000","1.300000","1.400000","1.500000","1.600000","1.700000","1.800000","2.000000"]:
    for lam in ["0.100000","0.300000","0.500000","0.700000","0.900000","1.000000","1.100000","1.300000","1.500000","1.700000","1.900000","2.100000","2.300000","2.500000","2.700000","2.900000"]:
        try:
            kappas = []
            kappaCsv = csv.reader(open(basepath + basename + kappa + middlepart + lam + ending))
            tmpKappa =[]
            for line in kappaCsv:
                for val in line:
                    if val != "":
                        tmpKappa += [float(val)]
                kappas += [np.mean(tmpKappa)]
            lams = []
            lamCsv = csv.reader(open(basepath + basenameLam + lam + middlepartLam + kappa + ending))
            tmpLam =[]
            for line in lamCsv:
                for val in line:
                    if val != "":
                        tmpLam += [float(val)]
                lams += [np.mean(tmpLam)]
            tmp = kappas
            kappas = lams
            lams = tmp 
            d = len(kappas) // (len(kappas)-1)
            ind = np.arange(d, len(kappas), d)
            for i in ind:
                try:
                    ar = FancyArrowPatch((kappas[i-1], lams[i-1]),(kappas[i], lams[i]), arrowstyle="->", mutation_scale=20)
                    ax.add_patch(ar)
                except StopIteration:
                    pass
            
            ax.plot(kappas,lams, "ro")
            ax.plot(kappas[0], lams[0], "b*")
            ax.plot(kappas[-1], lams[-1], "g*")
            plt.ylabel(r"$A(\kappa)$")
            plt.xlabel(r"$B(\lambda)$")
            plt.legend()
        except Exception:
            print("error: k=" + str(kappa) + "  l=" + str(lam))
plt.show()
