import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import csv

xs = [0.1,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.25,0.3,0.35,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47, 0.48, 0.49]

theCsv = csv.reader(open("./scripts/data/random_tests/scaling_test2.csv"))

ysmean =[]
yserror = []
for line in theCsv:
    tmp = []
    for m in line:
        if m == "":
            continue
        tmp += [float(m)]
    ysmean += [np.mean(tmp)]
    yserror += [np.std(tmp)/np.mean(tmp)]
yserror = np.array(yserror)
yserror /= np.sqrt(500)
plt.errorbar(xs, ysmean, yerr=yserror, capsize=3, fmt="ro", label="MC result / Theoretical Result")
plt.xlabel("$\kappa$")
plt.ylabel(r"$\frac{\left<\phi_i\phi_{i+2}\right>}{\left<\phi_i\phi_{i+4}\right>}$ deviation from theory")
plt.savefig("./data/phi4/nn_nnnn_mc.png")