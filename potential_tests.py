import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,v1,v2, kappa, lamb):
    return np.exp(-(x-kappa*(v1+v2))**2 - lamb * (x**2 -1)**2)
def mean(x,kappa,lamb):
    return np.exp(-(1-2*kappa)*x**2 - lamb*(x**2-1)**2)
def gauss(x,mean,var):
    return np.exp(-(1.0/(2*var**2)) * (x- mean)**2)
for l in [0.1]:
    for k in [0.3,0.5,1,5]:
        lamb = l
        kappa = k
        xs = np.linspace(-2,2,1000)
        vs = []
        while len(vs) < 2000:
            tmpgaus = np.random.normal(0, 1/np.sqrt(2))
            prop = np.min((1.0,mean(tmpgaus,k,l)/gauss(tmpgaus,0,1/np.sqrt(2))))
            p = np.random.uniform(0,1,1)
            if p < prop:
                vs += [tmpgaus]
        v1 = vs[:1000]
        v2 = vs[1000:]
        #print(v1,v2)
        ys = np.array([])
        ys = func(xs,v1[0],v2[0],kappa, lamb)
        for i in range(1,len(v1)):
            ys += func(xs,v1[i],v2[i],kappa, lamb)
            #plt.plot(xs, func(xs,v1[i],v2[i],kappa, lamb), "b-")
        ys /= len(v1)
        plt.plot(xs, ys, label=str(k))
        popt, cov = curve_fit(mean,xs, ys)
        print(popt)
        #plt.plot(xs,mean(xs,popt[0],popt[1]))
        ys2 = mean(xs,kappa,lamb)
        ys2 /= np.max(ys2)
        plt.plot(xs,ys2, "r-")
        plt.legend()
        plt.show()
