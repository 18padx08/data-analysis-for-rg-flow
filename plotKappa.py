import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-0.6,0.6,1000)
f = lambda x: 2*x**3 + x**2 -x
f2 = lambda x: x**2/(1-2*x**2)
f3 = lambda x:x**2/(1-2.09*x**2)
ys = f(x)
plt.plot(x,ys, label="kappa_hidden = f(kappa)")
plt.plot(x,0*x, "r-")
plt.plot(x,f2(x), "b-")
plt.plot(x,f3(x), "g-")

plt.xlabel(r"$\kappa$")
plt.ylabel(r"$\kappa^{(h)}$")
plt.legend()
plt.savefig("./data/phi4/various_functions.png")

plt.clf()

x=np.linspace(-1,1,100000)
plt.plot(x,f2(x), "b-", label=r"$\frac{\kappa^2}{1-2\kappa^2}$")
plt.plot(x,f3(x), "g-", label=r"$\frac{\kappa^2}{1-2.23\kappa^2}$")
plt.xlabel(r"$\kappa$")
plt.ylabel(r"$\kappa^{(h)}$")
plt.legend()
plt.savefig("./data/phi4/asymptotes.png")