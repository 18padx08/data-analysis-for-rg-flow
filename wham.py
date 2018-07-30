import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import re

def calculateErrorB4(theSamples, Z_s, newK, theObs):
    setSize = 5
    #generateNewBlockedData
    new = {}
    for key in theSamples.keys():
        #for each key we divide the 100 configurations in 20 blocks
        new[key] = []
        for i in range(int(100/setSize)):
            new[key] += [[]]
            for j in range(5):
                new[key][i] += [theSamples[key][j*20+i]]
                #print(new[key][i])
    #now calcualte partition funciton
    theZ = []
    obs = []
    for m in range(int(100/setSize)):
        theZ += [0]
        obs += [0]
        for kap in new.keys():
            counter =0 
            for block in new[kap]:
                #ignore block m
                if counter == m:
                    counter += 1
                    continue
                counter += 1
                for c in block:
                    innerValue = 0
                    for kj in new.keys():
                        innerValue += functionConstant(c[3], float(kj), Z_s[kj])
                    theZ[m] += np.exp((c[3])*2 * newK )/innerValue
                    obs[m] += np.exp((c[3]) * 2 * newK)/innerValue * (1-1.0/3.0*c[2]/c[1]**2)
        obs[m] /= theZ[m]
        obs[m] -= theObs
        obs[m] = obs[m] * obs[m]
    error = np.sqrt( (100/setSize -1)/(100/setSize) * np.sum(obs,0))
    return error


def Zk(theSamples, Z_s,newK):
    tmp = 0
    for kap in theSamples.keys():
        for config in theSamples[kap]:
            innerValue = 0
            for kj in theSamples.keys():
                innerValue += functionConstant(config[3],float(kj),Z_s[kj])
            #print(config[3] - avgEnergy)
            tmp += np.exp((config[3]) * 2 *newK)/innerValue
    return tmp

def phi4Obs(theSamples, Z_s,newK):
    tmp = 0
    for kap in theSamples.keys():
        for config in theSamples[kap]:
            innerValue = 0
            for kj in theSamples.keys():
                innerValue += functionConstant(config[3],float(kj),Z_s[kj])
            tmp += np.exp((config[3]) * 2 *newK)/innerValue * config[2]
    return tmp

def phi2Obs(theSamples, Z_s, newK):
    tmp = 0
    for kap in theSamples.keys():
        for config in theSamples[kap]:
            innerValue = 0
            for kj in theSamples.keys():
                innerValue += functionConstant(config[3],float(kj),Z_s[kj])
            tmp += np.exp((config[3]) * 2 *newK)/innerValue * config[1]
    return tmp

def phiAvg(theSamples, Z_s,newK):
    tmp = 0
    for kap in theSamples.keys():
        for config in theSamples[kap]:
            innerValue = 0
            for kj in theSamples.keys():
                innerValue += functionConstant(config[3],float(kj),Z_s[kj])
            tmp += np.exp((config[3]) * 2 *newK)/innerValue * config[0]
    return tmp

def weightedContrib(configuration, kappa, k_j, z_j):
    #print(configuration[3])
    value = 100.0/z_j * np.exp(2*(kappa - k_j) * configuration[3])
    return value

def functionConstant(action, k_j,z_j):
    myvalue = 100.0/z_j * np.exp(2*k_j*action)
    if(myvalue<1e-20):
        myvalue = 1e-20
    
    #print(tmp)
    return myvalue
colors = ["red", "blue", "orange", "green", "gray", "black", "yellow", "purple"]
colorCounter = 0
kappaReg = r".*kappa=(?P<kappa>\d+\.\d+).*"

for folder in ["8_8","12_12", "16_16"]:
    basepath = "./scripts/data/phi4_2d/wham/" + folder
    allsamples = {}
    Zs = {}
    for root, dirs, files in os.walk(basepath):
        if "zs.npy" in files:
            print("gotya")
            Zs = np.load(os.path.join(root,"zs.npy")).item()
        for file in files:
            if ".csv" not in file:
                continue
            kMatch = re.match(kappaReg, file)
            kappa = kMatch.group("kappa")
            theCsv = csv.reader(open(os.path.join(root,file)))
            allsamples[kappa] = []
            for line in theCsv:
                #absAvg, phi^2, phi^4, actionPart
                allsamples[kappa] += [[np.longdouble(line[0]), np.longdouble(line[1]), np.longdouble(line[2]), np.longdouble(line[3])]]
    #we have all samples sorted according to the respective kappas
    #now we need to iterate the partion functions until convergence
    
    Zsold = {}
    #initialize all zs to 1
    avgEnergy = 0
    zsExisted = len(Zs.keys())>0
    for k in allsamples.keys():
        if not zsExisted:
            Zs[k] = 100
            Zsold[k] = 100
        for config in allsamples[k]:
            avgEnergy += config[3]
    avgEnergy /= len(allsamples.keys()) * 100
    #avgEnergy= 0
    for kap in allsamples.keys():
        for config in allsamples[kap]:
            config[3] -= avgEnergy
    print(avgEnergy)
    print(Zs)
    finished = zsExisted
    print(finished)
    counter = 1
    threshold = 5e-2
    while not finished:
        #iterate over all zs and assign new values
        tmpBool = True
        tmpZs = {}
        for kappa in allsamples.keys():
            #print(Zs[kappa],Zsold[kappa])
            Zsold[kappa] = Zs[kappa]
            tmp = 0
            Zs[kappa] = Zk(allsamples,Zs,float(kappa))
            if abs(Zs[kappa] - Zsold[kappa]) < threshold * (Zs[kappa] + Zsold[kappa])/2.0:
                #print(tmpZs[kappa], Zsold[kappa])
                tmpBool = tmpBool and True
            else:
                tmpBool = tmpBool and False
        for kappa in allsamples.keys():
            #tmpZs[kappa] = tmpZs[kappa]
            pass
        finished = tmpBool
        if counter % 100 == 0:
           print(Zs)
           print(Zsold)
           print("")
        counter += 1
    print(Zs)
    #we have converged
    np.save(os.path.join(root,"zs.npy"), Zs)
    allphi4s = {}
    #interpolated points
    kappas = np.linspace(0.0,0.7,70)
    theZk = Zk(allsamples,Zs,kappas)
    thephi4s = phi4Obs(allsamples,Zs, kappas)/theZk
    thephi2s = phi2Obs(allsamples, Zs, kappas)/theZk

    ys = 1.0- 1.0/3.0 *thephi4s/thephi2s**2
    errors = calculateErrorB4(allsamples,Zs, kappas, ys)
    #plt.errorbar(kappas,ys,errors, capsize=2)
    plt.fill_between(kappas,ys -errors, ys+ errors, alpha=0.2, color=colors[colorCounter])
    plt.plot(kappas,ys, "-.", label=folder, color=colors[colorCounter])

    #stuetzpunkte
    
    allconfigs = {}
    for key in allsamples.keys():
        allphi4s[key] = 0
        allconfigs[key] = 0
        tmpconfigs = []
        for config in allsamples[key]:
            allphi4s[key] += 1 - 1.0/3.0 *config[2]/config[1]**2
            tmpconfigs += [1 - 1.0/3.0 *config[2]/config[1]**2]
        allphi4s[key] /= len(allsamples[key])
        allconfigs[key] = np.std(tmpconfigs)

    plt.plot([float(k) for k in allsamples.keys()], allphi4s.values(), "o", color=colors[colorCounter], label=folder)
    plt.errorbar([float(k) for k in allsamples.keys()],allphi4s.values(),allconfigs.values(), color=colors[colorCounter], capsize=2, fmt="o")
        #print(key, colors[colorCounter])
    colorCounter += 1
    
plt.xlabel(r"$\kappa$")
plt.ylabel(r"$1- \frac{<\phi^4>}{3<\phi^2>}$")
plt.title(r"$\lambda = 1$")
plt.legend()
plt.show()