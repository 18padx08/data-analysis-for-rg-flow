import numpy as np
import matplotlib.pyplot as plt

def gausses(x,vi,vi2):
    return (np.exp(-(x-2*0.4*vi)**2),np.exp(-(x-2*0.4*vi2)**2), np.exp(-2*(x-0.4*(vi+vi2))**2))

x = np.linspace(-2.5,2.5,10000)
vi = 1
vi2=1
a,b,c = gausses(x, vi,vi2)
#plt.plot(x,a, "r-")
#plt.plot(x,b, "b-")
#plt.plot(x,c, "g-")
#plt.plot(x,np.exp(2)/2.0* (a*b), "y-")
#plt.show()
first = np.random.normal(2*0.4*vi,1,(10000,1))
second = np.random.normal(2*0.4*vi2, 1.0, (10000,1))
comb = np.random.normal(0.4*(vi+vi2),1.0/np.sqrt(2), (10000,1))
plt.hist((first+second)/2, bins=100, color="blue")
plt.hist(comb, alpha=0.5, bins=100, color="red")
plt.show()
