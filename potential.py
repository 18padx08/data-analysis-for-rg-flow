import numpy as np
import matplotlib.pyplot as plt

def func(x, lamb, kappa):
    return  np.exp(-(1-2*kappa)*x**2 -lamb*(x**2 -1)**2)

xs = np.linspace(-4,4,1000)
for l in [1]:
    for k in [0.01]:
        plt.plot(xs, func(xs,l,k), label="lam=" + str(l) + ",kappa="+str(k))

plt.xlabel(r"$x$")
plt.ylabel(r"$V(\phi)$")
#plt.ylim(-5,0)
plt.legend()
plt.show()