import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import csv

xs = [0.1,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.25,0.3,0.35,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47, 0.48, 0.49]

theCsv = csv.reader(open("./scripts/data/random_tests/scaling_test.csv"))
mcnn = []
mcth = []
nnnmcth = []
nnth = []
nnnth = []
def f(x,slope,exp, xmean):
    return  slope*(10*x - 3)**exp
for line in theCsv:
    mcnn += [float(line[0])]
    mcth += [float(line[1])]
    nnth += [float(line[2])]
    nnnmcth += [float(line[3])]
    nnnth += [float(line[4])]

nnth = np.array(nnth)
nnnth = np.array(nnnth)
mcth = np.array(mcth)
nnnmcth = np.array(nnnmcth)
xs = np.array(xs)
#plt.plot(xs,nnth, "bo")
plt.plot(xs, mcth, "ro")

contx = np.linspace(np.min(xs), np.max(xs), 10000)

plt.xlabel("$\kappa$")
plt.ylabel("deviation to theoretical value in percent")
plt.legend()
plt.savefig("./data/phi4/scaling_vs_kappa_mc.png")
plt.clf()
blub1 = nnth/nnnth
blub3 = (nnth[5:] *np.exp(-2*(1.0/xs[5:] -2)))**2/np.exp(-4*(1.0/xs[5:] -2))
blub2 = mcth/nnnmcth
#plt.plot(xs, blub1, "go")
plt.xlabel("$\kappa$")
plt.ylabel(r"$\frac{\left<\phi_i\phi_{i+2}\right>}{\left<\phi_i\phi_{i+4}\right>}$ deviation from theory")
plt.plot(xs,blub2 , "yo")

plt.savefig("./data/phi4/nn_nnnn_mc.png")