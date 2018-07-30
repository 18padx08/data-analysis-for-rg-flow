import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import FancyArrowPatch
import csv
from scipy.optimize import curve_fit

def gauss(x, A, mu, sigma):
    #A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


for bs in [10,20,40,80]:
    #[("0.6","0_6"),("0.8","0_8"),("1.0","1_0"),("1.2","1_2"),("1.4","1_4")]
    for c in [("0.6","0_6"),("0.8","0_8"),("1.0","1_0"),("1.2","1_2"),("1.4","1_4")]:
        couplings = np.zeros((200,5))
        betaj =c[0] + "00000"
        bj = c[1]
        batchSize = bs
        chainSize = 512
        file = open("scripts/data/rg_flow_" + str(bj) + "_2/rg_flow_comb_betaj="+betaj+"_bs=" + str(batchSize) + "_cs="+str(chainSize)+".csv")
        csvData = csv.reader(file, delimiter=",")
        j=0
        for line in csvData:
            ele = line
            i = 0
            for el in ele:
                if i >= len(line)-1:
                    continue
                couplings[i][j] = float(el)
                i += 1
            j += 1
        maxes = []
        variances = []
        couplings = couplings.T
        for i in range(0,5):
            try:
                hist, bin_edges = np.histogram(couplings[i],normed=True)
                bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
                p0 = [np.max(hist),bin_centres[np.argmax(hist)],np.std(couplings[i])]
                coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
                x = np.linspace(bin_centres[0], bin_centres[-1], 1000)
                hist_fit = gauss(x, *coeff)
                plt.plot(bin_centres, hist, "ro", label='Test data')
                plt.plot(x, hist_fit, label='Fitted data')
                baba = np.invert(np.isfinite(var_matrix))
                if len(var_matrix[baba]) <= 0:
                    maxes += [np.abs(coeff[1])]
                    variances += [coeff[2]]
                else:
                    maxes += [np.abs(np.mean(couplings[i][:]))]
                    variances += [np.std(np.tanh(couplings[i][:]))]
                plt.xlabel("A")
                plt.ylabel("#events")
                plt.legend()
                plt.savefig("./data/histogram_bj=" + str(bj) +"_bs=" + str(batchSize) + "_cs=" + str(chainSize) + "_layer=" + str(i) +"_auto_3.png")
                plt.clf()
                #plt.show()
            except RuntimeError:
                print("runtime error")
                maxes += [np.abs(np.mean(couplings[i][:]))]
                variances += [np.std(np.tanh(couplings[i][:]))]
        bluber =np.tanh(maxes)
        plt.errorbar(np.tanh(maxes), np.tanh(maxes),variances, variances,"go", capsize=5)
        #zero = np.abs(np.mean(couplings[-1][:]))
        #plt.plot(np.tanh(zero), np.tanh(zero), "ro")
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
        ax = plt.gca()
        for i in range(0, len(bluber)):
            #plt.plot([bluber[i],bluber[i]], [bluber[i], f(bluber[i])], "r-.")
            ar = FancyArrowPatch([bluber[i],bluber[i]], [bluber[i], f(bluber[i])], arrowstyle="->",linestyle="dashed", mutation_scale=20, color="red")
            ax.add_patch(ar)
            #plt.plot([bluber[i],f(bluber[i])], [f(bluber[i]), f(bluber[i])], "r-.")
            ar2 = FancyArrowPatch([bluber[i],f(bluber[i])], [f(bluber[i]), f(bluber[i])], arrowstyle="->",linestyle="dashed", mutation_scale=20, color="red")
            ax.add_patch(ar2)
        #plt.show()
        plt.text(0,0.9,r"$\beta J =" + str(betaj) + r"$")
        plt.text(0,0.85, r"$batchsize = " + str(batchSize) + r"$")
        plt.savefig("./data/rg_flow_bj=" + str(bj) +"_bs=" + str(batchSize) + "_cs=" + str(chainSize) + "_auto_3.png")
        plt.clf()