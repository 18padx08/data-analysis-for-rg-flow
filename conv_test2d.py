import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import re

reg = r".*kappa=(?P<kappa>\d+\.\d+?)_lamb=(?P<lambda>\d+\.\d+?)_cs=(?P<cs>\d+?)bs=(?P<bs>\d+?)_lr=(?P<lr>\d+\.\d+?)\.csv"

basepath = "./scripts/data/phi4_2d/conf_test/conv_test/2d_conv_test"
for root, dirs, files in os.walk(basepath):
    for file in files:
        if ".csv" not in file:
            continue
        theCsv = csv.reader(open(os.path.join(root,file)))
        kappas = []
        lambdas = []
        for line in theCsv:
            kappas += [float(line[0])]
            lambdas += [float(line[1])]
        fig, ax1 = plt.subplots()
        ax1.plot(kappas, "ro")
        ax1.hlines(kappas[-1],-10, len(kappas)+10,"red","solid")
        ax1.set_xlabel("mc time")
        ax1.tick_params("y", colors="r")
        ax1.set_xlim(0, len(kappas))
        ax2 = ax1.twinx()
        ax2.plot(lambdas, "bo")
        ticks = np.array(list(ax1.get_yticks()))
        
        print(type(ticks[0]))
        #print(np.abs(ticks - np.mean(kappas[-10:])))
        ticks =  ticks[np.abs(ticks - np.mean(kappas[-10:])) > 0.2 * (ticks[1]-ticks[0])]
        ticks = np.append(ticks, [np.mean(kappas[-10:])])
        ax1.set_yticks(ticks)

        ax1.set_ylabel(r"$\kappa$", color="r")
        ax2.set_ylabel(r"$\lambda$", color="b")
        ax2.tick_params("y", colors="b")
        ax2.set_xlim(0, len(lambdas))
        ax2.hlines(lambdas[-1], -10, len(lambdas)+10,"blue", "solid")
        ticks = np.array(ax2.get_yticks())
        ticks =  ticks[np.abs(ticks - np.mean(lambdas[-10:])) > 0.2 * (ticks[1]-ticks[0])]
        ticks = np.append(ticks, [np.mean(lambdas[-10:])])
        ax2.set_yticks(ticks)
        fig.tight_layout()
        match = re.match(reg,file)
        title = ""
        if match is not None:
            title += r"$\kappa=" + str(np.round(float(match.group("kappa")),2)) + r"\quad \lambda=" + str(np.round(float(match.group("lambda")), 2) )+ r"\quad L=" + match.group("cs") + r"\quad B_s=" + match.group("bs") + r"\quad \eta=" + str(np.round(float(match.group("lr")), 2) ) +r"$"
        plt.title(title)
        ax1.grid(True)
        #ax2.grid(True)
        plt.legend()
        plt.savefig("./data/phi4_2d/convergence/" + file + ".png")
        #plt.show()
