import numpy as np
import matplotlib.pyplot as plt

maxes = [0.981439,0.643811,0.36434,0.136199]

bluber =np.tanh(maxes)
plt.plot(np.tanh(maxes), np.tanh(maxes), "ro")
plt.rc('text', usetex=True)
plt.ylabel(r'$tanh(\beta J)$')
plt.xlabel(r'$tanh(\beta J)$')
x = np.linspace(1,0,100)
y = np.tanh(np.arctanh(x))**2
f = lambda x: np.tanh(np.arctanh(x))**2
interpol = np.tanh(maxes)
params= np.polyfit(interpol, interpol, 1)

plt.plot(x,y, "b")
plt.plot(x,params[0]*x + params[1], "b")

#plot connections
line1 = []
print(line1)
for i in range(0, len(bluber)):
    plt.plot([bluber[i],bluber[i]], [bluber[i], f(bluber[i])], "r-.")
    plt.plot([bluber[i],f(bluber[i])], [f(bluber[i]), f(bluber[i])], "r-.")
plt.show()