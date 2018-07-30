import numpy as np
import matplotlib.pyplot as plt
import csv

theCsv = csv.reader(open("./scripts/data/susceptability.csv"))
xs =[]
ys = []
for line in theCsv:
    xs += [float(line[0])]
    ys += [float(line[1])]

plt.plot(xs,ys, "ro")
plt.xlabel("kappa")
plt.ylabel("$<\phi>$")
plt.rc('text', usetex=True)
plt.show()

theCsv = csv.reader(open("./scripts/data/avgAbsPhi.csv"))
xs =[]
ys = []
for line in theCsv:
    xs += [round(float(line[0]),4)]
    ys += [round(float(line[1]),4)]

plt.plot(xs,ys, "ro")
plt.xlabel("kappa")
plt.ylabel("$<|\phi|>$")
plt.rc('text', usetex=True)
plt.show()