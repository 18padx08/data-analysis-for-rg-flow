import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

clusterGuesses = []
fftGuesses = []
c = csv.reader(open("./scripts/data/phi4/cluster_vs_fft_cs=512_kappa=0.4.csv"))
for line in c:
    clusterGuesses += [float(line[0])]
    fftGuesses += [float(line[1])]
def gauss(x, A,sigma, mean):
    return A*np.exp(-(x-mean)**2/(2*sigma**2))


clusterGuesses = np.array(clusterGuesses)
fftGuesses = np.array(fftGuesses)
limits = [np.min([np.min(clusterGuesses), np.min(fftGuesses)]),np.max([np.max(clusterGuesses), np.max(fftGuesses)])]
hist, bin_edges = np.histogram(clusterGuesses, bins=100,range=limits)

bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

p0 = [3000,0.4,0.2]
coeff, var_matrix = curve_fit(gauss, bin_centres, hist)
print(coeff)
x = np.linspace(bin_centres[0], bin_centres[-1], 1000)
hist_fit = gauss(x, *coeff)
plt.hist(clusterGuesses, bins=100, label='Test data Metropolis-Hasting')
plt.plot(x, hist_fit, label='Fitted data Metropolis-Hasting')

hist, bin_edges = np.histogram(fftGuesses, bins=100, range=limits)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
p0 = [3000,0.4,0.2]
coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
print(coeff)
x = np.linspace(bin_centres[0], bin_centres[-1], 1000)
hist_fit = gauss(x, *coeff)

plt.hist(fftGuesses, bins=100, label='Test data fft', alpha=0.5)
plt.plot(x, hist_fit, label='Fitted data fft')
plt.legend()
plt.show()


clusterGuesses = []
fftGuesses = []
c = csv.reader(open("./scripts/data/phi4/vis_hid_renorm_kappa_cs=512_kappa=0.45.csv"))
for line in c:
    clusterGuesses += [float(line[0])]
    fftGuesses += [float(line[1])]
def gauss(x, A,sigma, mean):
    return A*np.exp(-(x-mean)**2/(2*sigma**2))


clusterGuesses = np.array(clusterGuesses)
fftGuesses = np.array(fftGuesses)
limits = [np.min([np.min(clusterGuesses), np.min(fftGuesses)]),np.max([np.max(clusterGuesses), np.max(fftGuesses)])]
hist, bin_edges = np.histogram(clusterGuesses, bins=100, range=limits)
print("area of cluster")
print(np.sum(hist))
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

p0 = [3000,0.4,0.2]
coeff, var_matrix = curve_fit(gauss, bin_centres, hist)
print(coeff)
x = np.linspace(bin_centres[0], bin_centres[-1], 1000)
hist_fit = gauss(x, *coeff)
plt.hist(clusterGuesses, bins=100, label='Test data original', range=limits)
plt.plot(x, hist_fit, label='Fitted data original')

hist, bin_edges = np.histogram(fftGuesses, bins=100, range=limits)
print("area of fft")
print(hist.shape)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
p0 = [3000,0.4,0.2]
coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
print(coeff)
x = np.linspace(bin_centres[0], bin_centres[-1], 1000)
hist_fit = gauss(x, *coeff)

plt.hist(fftGuesses, bins=100, label='Test data renormalized', alpha=0.5, range=limits)
plt.plot(x, hist_fit, label='Fitted data renormalized')
plt.legend()
plt.show()