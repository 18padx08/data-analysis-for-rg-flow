import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import csv

variances = []
chainSizes =[16,32,64,128,256,512,1024]
for cs in chainSizes:
    theCsv = csv.reader(open("./scripts/data/random_tests/variance_tests/variance_test_" +str(cs) + "_.csv"))
    tmpvar = []
    for line in theCsv:
        tmpvar += [float(line[0])]
    tmpvar = np.array(tmpvar)
    variances += [np.var(tmpvar)]

def decay(x, expo, slope):
    return slope / x**expo
plt.plot(chainSizes, variances, "ro")
popt,cov =scipy.optimize.curve_fit(decay, chainSizes, variances, p0=[1.0/2.0,np.sqrt(2)])
xs = np.linspace(16,2000,10000)
plt.plot(xs, decay(xs, popt[0],popt[1]))
print(popt)
plt.xlabel("chain size")
plt.ylabel("variance")
plt.legend()
plt.show()