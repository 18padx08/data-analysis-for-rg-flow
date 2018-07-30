import numpy as np
import matplotlib.pyplot as plt
import csv

theCsv = csv.reader(open("./scripts/data/check_randomness/gauss_logs_lam=10_mean=-1.csv"))
samples = []
i = 0
for line in theCsv:
    tmp = []
    for num in line:
        if num != "":
            tmp += [float(num)]
    samples += [tmp]
    i += 1
    if i > 30:
        break

totalmeans = 0
totalvar = 0
for ds in samples:
    plt.hist(ds, bins=100)
    totalmeans+=np.mean(ds)
    totalvar += np.var(ds)
totalmeans /= 30
totalvar /= 30
print(totalmeans, totalvar)
plt.show()

