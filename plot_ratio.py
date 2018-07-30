import numpy as np
import matplotlib.pyplot as plt

def newKappa(x):
    return 1.0/(1.0/(4*x) + 3.0/2.0)

kappa = np.linspace(0.1,0.5,1000)
blub = (2*kappa / newKappa(kappa))

plt.plot(kappa, 4*kappa**4-2*kappa**2 +kappa**(1.0/2.0), "r-")
plt.show()