import csv
import numpy as np
import matplotlib.pyplot as plt

theCsv = csv.reader(open("./scripts/data/random_tests/norm_tests.csv"), delimiter=",")
th =0
mc = []
nn = []
for line in theCsv:
    th = line[0]
    mc += [line[1]]
    nn += [line[2]]

mc = np.array(mc, dtype=float)

nn = np.array(nn, dtype=float)
nn = nn 
theBins = np.linspace(0.19,0.3,30)
plt.hist(mc, theBins, color="blue", alpha = 0.5)

plt.hist(nn,theBins, color = "red", alpha = 0.5)
plt.show()
print(np.mean(mc), np.mean(nn), np.std(mc), np.std(nn), np.var(nn), np.std(nn)**2)
