import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import re

configurationSize = 100

def calculateErrorB4(theSamples, Z_s, newK, theObs):
    setSize = 10
    #generateNewBlockedData
    new = {}
    theZss = []
    for key in theSamples.keys():
        #for each key we divide the 100 configurations in 20 blocks
        new[key] = []
        numberOfBlocks = int(configurationSize/setSize)
        for i in range(numberOfBlocks):
            new[key] += [[]]
            for j in range(setSize):
                new[key][i] += [theSamples[key][j*numberOfBlocks+i]]
                #print(new[key][i])
    #now calcualte partition funciton
    theZ = []
    obs = []
    Zsold = {}
    for m in range(int(configurationSize/setSize)):
        print("solving system of equations for block " + str(m))
        newZs = solveNonLinEqRemoveBlockM(new,Z_s,m)
        print("calculate the new observable")
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
                        actionPart = config[3]
                        #use this for lambda analysis
                        #actionPart = (config[1]-1)**2
                        innerValue += functionConstant(actionPart, float(kj) - newK, newZs[kj])
                    theZ[m] += 1.0/innerValue #np.exp((c[3])*2 * newK )/innerValue
                    obs[m] += 1.0/innerValue * np.abs(c[0])#(1-1.0/3.0*c[2]/c[1]**2) #np.exp((c[3]) * 2 * newK)/innerValue * (1-1.0/3.0*c[2]/c[1]**2)
        obs[m] /= theZ[m]
        obs[m] -= theObs
        obs[m] = obs[m] * obs[m]
        print("go to next m")
    error = np.sqrt( (configurationSize/setSize -1)/(configurationSize/setSize) * np.sum(obs,0))
    return (error, theZss)

def ZkMRemoved(theSamples, Z_s,newK, m):
    tmp = 0
    for kap in theSamples.keys():
        counter = 0
        for block in theSamples[kap]:
            if counter == m:
                counter += 1
                continue
            counter += 1
            for config in block:
                innerValue = 0
                for kj in theSamples.keys():
                    actionPart = config[3]
                    #use this for lambda analysis
                    #actionPart = (config[1]-1)**2
                    innerValue += functionConstant(actionPart,float(kj) - newK,Z_s[kj])
                    
                #print(config[3] - avgEnergy)
                tmp += 1.0/innerValue #np.exp((config[3]) * 2 *newK)/innerValue
    return tmp
def Zk(theSamples, Z_s,newK):
    tmp = 0
    for kap in theSamples.keys():
        for config in theSamples[kap]:
            innerValue = 0
            for kj in theSamples.keys():
                actionPart = config[3]
                #use this for lambda analysis
                #actionPart = (config[1]-1)**2
                innerValue += functionConstant(actionPart,float(kj) - newK,Z_s[kj])
            #print(config[3] - avgEnergy)
            tmp += 1.0/innerValue  #np.exp((config[3]) * 2 *newK)/innerValue
    return tmp

def phi4Obs(theSamples, Z_s,newK):
    tmp = 0
    for kap in theSamples.keys():
        for config in theSamples[kap]:
            innerValue = 0
            for kj in theSamples.keys():
                actionPart = config[3]
                #use this for lambda analysis
                #actionPart = (config[1]-1)**2
                innerValue += functionConstant(actionPart,float(kj) - newK,Z_s[kj])
            tmp += 1.0/innerValue *config[2] #np.exp((config[3]) * 2 *newK)/innerValue * config[2]
    return tmp

def phi2Obs(theSamples, Z_s, newK):
    tmp = 0
    for kap in theSamples.keys():
        for config in theSamples[kap]:
            innerValue = 0
            for kj in theSamples.keys():
                actionPart = config[3]
                #use this for lambda analysis
                #actionPart = (config[1]-1)**2
                innerValue += functionConstant(actionPart,float(kj) - newK,Z_s[kj])
            tmp += 1.0/innerValue*config[1] #np.exp((config[3]) * 2 *newK)/innerValue * config[1]
    return tmp

def phiAvg(theSamples, Z_s,newK):
    tmp = 0
    for kap in theSamples.keys():
        for config in theSamples[kap]:
            innerValue = 0
            for kj in theSamples.keys():
                innerValue += functionConstant((config[1]-1)**2,float(kj)-newK,Z_s[kj])
            tmp += 1.0/innerValue * np.abs(config[0])
    return tmp

#def weightedContrib(configuration, kappa, k_j, z_j):
 #   #print(configuration[3])
  #  value = configurationSize/z_j * np.exp(2*(kappa - k_j) * configuration[3])
   # return value

def functionConstant(action, k_j,z_j):
    myvalue = configurationSize/z_j * np.exp(2*k_j*action)
    #if(len(myvalue[myvalue<1e-20]) > 0):
     #   myvalue[myvalue < 1e-20] = 1e-20
    
    #print(tmp)
    return myvalue

def solveNonLinEqRemoveBlockM(allsamples, Zs, m):
    threshold = 5e-2
    counter = 0
    finished = False
    Zsold = Zs.copy()
    newZs = Zs.copy()
    while not finished:
        #iterate over all zs and assign new values
        tmpBool = True
        tmpZs = {}
        for kappa in allsamples.keys():
            #print(Zs[kappa],Zsold[kappa])
            Zsold[kappa] = newZs[kappa]
            tmp = 0
            newZs[kappa] = ZkMRemoved(allsamples,newZs,float(kappa), m)
            
        #for kappa in allsamples.keys():
            #tmpZs[kappa] = tmpZs[kappa]
        #    pass
        theSum = 0
        blub = sorted(newZs.values())
        for val in blub:
            theSum += val
        for kap in newZs.keys():
            newZs[kap] /= theSum
        for kappa in allsamples.keys():
            if abs(newZs[kappa] - Zsold[kappa]) < threshold * (newZs[kappa] + Zsold[kappa])/2.0:
                #print(tmpZs[kappa], Zsold[kappa])
                tmpBool = tmpBool and True
            else:
                #print(abs(newZs[kappa] - Zsold[kappa]))
                #print(threshold * (newZs[kappa] + Zsold[kappa])/2.0)
                tmpBool = tmpBool and False
        finished = tmpBool
        #tmpBool = True
        if counter % 10 == 0:
            #print(newZs)
            #print(Zsold)
            #for kappa in allsamples.keys():
            #    if abs(newZs[kappa] - Zsold[kappa]) < threshold * (newZs[kappa] + Zsold[kappa])/2.0:
                    #print("blub")
            #        tmpBool = tmpBool and True
            #    else:
                    #print("not blub")
            #        tmpBool = tmpBool and False
          # print(m)
            print("")
        #print(tmpBool)
        finished = tmpBool
        counter += 1
    return newZs
def solveNonLinEq(allsamples, newZs, theZsOld,finished):
    threshold = 5e-3
    counter = 0
    Zs = newZs.copy()
    Zsold = theZsOld.copy()
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
        theSum = 0
        blub = sorted(Zs.values())
        for val in blub:
            theSum += val
        for kap in Zs.keys():
            Zs[kap] /= theSum
        if counter % 5 == 0:
           print(Zs)
           print(Zsold)
           print("")
        counter += 1
    return Zs
colors = ["red", "blue", "orange", "green", "gray", "black", "yellow", "purple"]
colorCounter = 0
kappaReg = r".*lambda=(?P<lamb>\d+\.\d+).*kappa=(?P<kappa>\d+\.\d+).*"

for folder in ["16_16", "20_20", "25_25","30_30", "32_32"]:
    first = True
    basepath = "./scripts/data/fss/fixed_kappa/" + folder
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
            if first:
                first = False
                configurationSize = len(allsamples[kappa])
    #we have all samples sorted according to the respective kappas
    #now we need to iterate the partion functions until convergence
    
    print(configurationSize)
    Zsold = {}
    #initialize all zs to 1
    avgEnergy = 0
    zsExisted = len(Zs.keys())>0
    for k in allsamples.keys():
        if not zsExisted:
            Zs[k] = 1.0/len(allsamples.keys())
            Zsold[k] = 1.0/len(allsamples.keys())
        for config in allsamples[k]:
            avgEnergy += config[3]
    if len(allsamples.keys()) <=  0:
        print(allsamples, Zs, file, folder,len(files))
    avgEnergy /= len(allsamples.keys()) * configurationSize
    #avgEnergy= 0
    for kap in allsamples.keys():
        for config in allsamples[kap]:
            config[3] -= avgEnergy
    #print(avgEnergy)
    #print(Zs)
    finished = zsExisted
    #print(finished)
    Zs = solveNonLinEq(allsamples, Zs, Zsold,finished)
    counter = 1
    #print(Zs)
    #we have converged
    np.save(os.path.join(root,"zs.npy"), Zs)
    allphi4s = {}
    #interpolated points
    kappas = np.linspace(0.01,0.2,500)
    theZk = Zk(allsamples,Zs,kappas)
    thephi4s = phi4Obs(allsamples,Zs, kappas)/theZk
    thephi2s = phi2Obs(allsamples, Zs, kappas)/theZk
    thePhiAvg = phiAvg(allsamples,Zs,kappas)/theZk
    np.save(os.path.join(root, "kappas.npy"), kappas)
    np.save(os.path.join(root,"phi4.npy"), thephi4s)
    np.save(os.path.join(root,"phi2.npy"), thephi2s)
    np.save(os.path.join(root,"phiAvg.npy"), thePhiAvg)
    ys = np.abs(thePhiAvg)#1.0- 1.0/3.0 *thephi4s/thephi2s**2
    errors, zss = calculateErrorB4(allsamples,Zs, kappas, ys)
   # plt.plot([firstZs] + zss, "ro")
    np.save(os.path.join(root,"errors.npy"), errors)
    np.save(os.path.join(root,"zsserrors.npy"), zss)
    #plt.show()
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
            allphi4s[key] += np.abs((config[1]-1)**2)#1 - 1.0/3.0 *config[2]/config[1]**2
            tmpconfigs += [np.abs((config[1]-1)**2)]# [1 - 1.0/3.0 *config[2]/config[1]**2]
        allphi4s[key] /= len(allsamples[key])
        allconfigs[key] = np.std(tmpconfigs)/np.sqrt(len(tmpconfigs))

    plt.plot([float(k) for k in allsamples.keys()], allphi4s.values(), "o", color=colors[colorCounter], label=folder)
    plt.errorbar([float(k) for k in allsamples.keys()],allphi4s.values(),allconfigs.values(), color=colors[colorCounter], capsize=2, fmt="o")
        #print(key, colors[colorCounter])
    colorCounter += 1
    
plt.xlabel(r"$\kappa$")
plt.ylabel(r"$<|\phi|>$")
plt.title(r"$\lambda = 0.1$")
plt.legend()
plt.show()