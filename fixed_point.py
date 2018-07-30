import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import csv

def rgFlow(x, c):
    return (x**2)/(1 - c*x**2)

basePath = "./scripts/data/phi4/rg_flow/"
baseName = "rg_flow_comb_betaj="
layers = []
layerMean = []
layerVar = []
means = []
theVars = []
realList = [ "0.100000", "0.200000","0.300000","0.350000","0.400000","0.420000","0.430000","0.450000","0.460000","0.470000","0.480000","0.490000" ]
overal = 0
for betaj in realList :
    theCsv = csv.reader(open(basePath+baseName+betaj+"_bs=20_cs=512.csv"))
    i = 0
   # means += [float(betaj)]
   # theVars += [0]
    for line in theCsv:
        layers += [[]]
        layerMean += [[]]
        layerVar += [[]]
        currentMean = []
        for part in line:
            if part != "":
                layers[i] += [float(part)]
                currentMean += [float(part)]
        theMean = np.mean(currentMean)
        if betaj == "0.440000":
            print(theMean)
        overal = 200
        means += [theMean]
        theVars += [np.std(currentMean)**2]
        i+=1


xs = []
ys = []
sizeOfList = 5
x = np.linspace(0,0.5,1000)
for i in range(len(realList)):
    index = i *sizeOfList
    plt.errorbar(means[index:index+sizeOfList-1],means[(index+1):index+sizeOfList],yerr=theVars[(index+1):index+sizeOfList],xerr=theVars[index:index+sizeOfList-1], fmt="o", label="data" + str(i))
    xs += means[index:index+sizeOfList-1]
    ys += means[(index+1):index+sizeOfList]
fit,cov = scipy.optimize.curve_fit(rgFlow, xs,ys,p0=(2))
plt.plot(x, rgFlow(x,*fit),"r-", label=r"$y = \frac{\kappa^2}{1-" + str(round(fit[0],2)) + r"\kappa^2}$")
print(fit)

#plt.show()
f = lambda oldCoupling: oldCoupling**2/(1-2*oldCoupling**2)

plt.plot(x,f(x), "b:", label="theoretical")
plt.legend()
plt.savefig("./data/phi4/rg_flow_error_phi4_bs=0.200000_cs=512_non_const_renorm.png")
