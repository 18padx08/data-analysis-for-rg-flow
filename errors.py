import numpy as np
import matplotlib.pyplot as plt
#cd for 1 lr=0.1; 5 gibbs steps
data = [0.419974,0.119974,0.0899743,0.0389743,0.0740743]
x = [1,10,100,1000,10000]
plt.rc('text', usetex=True)
plt.semilogx(x,data, "bo")
contX = np.logspace(0,4)
y = 1.0 / np.sqrt(contX)
plt.semilogx(contX, y, label=r'$\frac{1}{\sqrt{x}}$')
plt.semilogx(50,1.0/np.sqrt(50), "ro", label="50")
plt.xlabel("#samples")
plt.ylabel(r"$\Delta x$")
plt.legend(loc="best")
plt.show()

data = [0.419974,0.119974,0.0900257,0.00897434,0.0652743]
x = [1,10,100,1000,10000]
plt.rc('text', usetex=True)
plt.semilogx(x,data, "bo")
contX = np.logspace(0,4)
y = 1.0 / np.sqrt(contX)
plt.semilogx(contX, y, label=r'$\frac{1}{\sqrt{x}}$')
plt.semilogx(500,1.0/np.sqrt(500), "ro", label="500")
plt.xlabel("#samples")
plt.ylabel(r"$\Delta x$")
plt.legend(loc="best")
plt.show()


data = [1.58003,0.0800257,0.0500257,0.0229743,0.0612743]
x = [1,10,100,1000,10000]
plt.rc('text', usetex=True)
plt.semilogx(x,data, "bo")
contX = np.logspace(0,4)
y = 1.0 / np.sqrt(contX)
plt.semilogx(contX, y, label=r'$\frac{1}{\sqrt{x}}$')
plt.semilogx(50,1.0/np.sqrt(50), "ro", label="50")
plt.xlabel("#samples")
plt.ylabel(r"$\Delta x$")
plt.legend(loc="best")
plt.show()


#100 gibbs steps lr=0.1; cd for 10
data = [0.419974,0.319974,0.169974,0.0989743,0.0608743]
x = [1,10,100,1000,10000]
plt.rc('text', usetex=True)
plt.semilogx(x,data, "bo")
contX = np.logspace(0,4)
y = 1.0 / np.sqrt(contX)
plt.semilogx(contX, y, label=r'$\frac{1}{\sqrt{x}}$')
plt.semilogx(50,1.0/np.sqrt(50), "ro", label="50")
plt.xlabel("#samples")
plt.ylabel(r"$\Delta x$")
plt.legend(loc="best")
plt.show()