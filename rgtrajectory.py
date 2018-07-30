import numpy as np
import matplotlib.pyplot as plt
import csv
import os
ys = []
file = open("scripts/data/coupling_traj_1.400000_cheatCD.csv")
csvreader = csv.reader(file)
for line in csvreader:
    ys += [float(line[0])]
plt.plot(ys)
x = np.linspace(0, len(ys), len(ys))
y = [np.mean(ys[5:])] * len(ys)
plt.xlabel("iterations")
plt.ylabel("A")
plt.text(-25,y[0] -0.03, np.round(y[0], 2), color="red")
plt.plot(x,y,"r")
plt.show()