import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.optimize as opt
import lin_fit_conf_inter as lin_fit
import re

def mult_gaussFun_Fit(input_data,*m):
    (x,y) = input_data
    A,x0,y0,varx,vary = m
    X,Y = np.meshgrid(x,y)
    Z = A*np.exp(-(X-x0)**2/(2*(varx)**2)-(Y-y0)**2/(2*(vary)**2))
    return Z.ravel()

def mult_gauss_rot(input_data, *m):
    (x,y) = input_data
    A,varx,vary,rot_angle = m
    x0=0
    y0=0
    X,Y = np.meshgrid(x,y)
    a = np.cos(rot_angle)**2/(2*varx**2) + np.sin(rot_angle)**2 /(2*vary**2)
    b= -np.sin(2*rot_angle)/(4*varx**2) + np.sin(2*rot_angle)/(4*vary**2)
    c = np.sin(rot_angle)**2/(2*varx**2) + np.cos(rot_angle)**2/(2*vary**2)
    Z = A *np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) +c*(Y-y0)**2))
    return Z.ravel()

def mult_gauss_rot_abc(input_data, A, a,b,c, offset):
    (x,y) = input_data
    X, Y = np.meshgrid(x,y)
    Z = offset + A * np.exp(-(a*X**2 + 2.0*b*X*Y + c*Y**2))
    return Z.ravel()
slopes = {0.2:[], 0.3 :[],0.4:[], 0.45:[],0.48:[]}
slope_cov = {0.2:[], 0.3 :[],0.4:[], 0.45:[],0.48:[]}
for bs in [20,40]:
    for bj in ["0.400000"]:
        batch_size = bs
        betaj = bj
        #plot error in correlation length vs MC error
        xs = []
        ys =[]
        ys2 = []
        xs3 = []
        ys3 = []
        zs3 = []
        c = csv.reader(open("./scripts/data/phi4_fixedpoint/data/response_error"+str(betaj) +"_bs=" +str(batch_size)+"_cs=2048.csv"), delimiter=",")
        m = np.sqrt(1.0/float(bj) - 2)
        for line in c:
            xs += [(np.exp(-2*m) - float(line[3])) ]
            ys += [(np.exp(-2*m) - float(line[5]))]
            ys2 += [(np.exp(-2*m) - float(line[4]))]
            xs3 += [ float(line[3])]
            ys3 += [float(line[4])]
            zs3 += [float(line[5])]
        #don't use the ones from response_error, but use stdev
        As = []
        mc = []
        vCL = []
        hCL = []
        for i in range(0,200):
            cA = csv.reader(open("./scripts/data/phi4_fixedpoint/data/error_gauss_bj=" + str(betaj) + "_" +str(i)+"_bs=" + str(batch_size) + "_cs=2048.csv"))
            cMC = csv.reader(open("./scripts/data/phi4_fixedpoint/data/error_gauss_mc_bj=" + str(betaj) + "_" +str(i)+"_bs=" + str(batch_size) + "_cs=2048.csv"))
            #does have two entries
            cCL = csv.reader(open("./scripts/data/phi4_fixedpoint/data/error_gauss_nn_bj=" + str(betaj) + "_" +str(i)+"_bs=" + str(batch_size) + "_cs=2048.csv"))
            tmp = []
            for line in cA:
                tmp += [float(line[0])]
            As += [np.std(tmp)]
            tmp = []
            for line in cMC:
                tmp += [float(line[0])]
               # tmpstring = ""
               # for part in line:
                #stupid workaround
                #    tmpstring += part
                #stringIter = re.finditer(r"-?0\.\d*(?=(-?0\.|,))", tmpstring)
                #theStrings = []
                #for num,match in enumerate(stringIter):
                 #   theStrings += [float(match.group())]
                #tmp +=theStrings[::2]
            
            mc += [np.std(tmp)]
            tmp2 = []
            for line in cCL:
                #remember visible and hidden stored
                tmp += [float(line[0])]
                tmp2 += [float(line[1])]
            vCL += [np.std(tmp)]
            hCL += [np.std(tmp2)]

        #now xs is the visible correlation length ys is the monte carlo and ys2 is the hidden
        xs = np.array(xs, dtype=float)
        #print((1.0/np.cosh(xs)**2))
       # print(2.0 *np.tanh(xs) * 1.0/np.cosh(xs)**2)
       # xs = xs / 2
        vCL = np.array(vCL)
        ys= np.array(ys)
        mc = np.array(mc)
        #mc = mc * np.sqrt(2)
        #mc = mc - np.mean(mc)
        #mc = mc / (1.0/np.cosh(mc)**2)
        ys2 = np.array(ys2)
        #ys2 = ys2 /2
        hCL = np.array(hCL)

        xs = np.array(xs3, dtype=float)
        xs = xs
        #xs = xs - np.mean(xs)
        #ys2 = ys2 - np.mean(ys2)
        corrLenghtError = xs
        ys = np.array(zs3, dtype=float)
        #ys = ys * np.sqrt(2)
        #ys = ys - np.mean(ys)
        ys2 = np.array(ys2, dtype=float)
        ys2 = ys2
        #ys2 = ys2 - np.mean(ys2)
        hiddenCorrLengthError = ys2
        #fit the 2d gaussian
        xmin = -1/np.sqrt(batch_size)
        xmax = 1/np.sqrt(batch_size)
        ymin = -1/np.sqrt(batch_size)
        ymax =  1/np.sqrt(batch_size)
        limits = [[xmin,xmax], [ymin,ymax]]
        H, xedges, yedges = np.histogram2d(xs,ys, bins=60, normed=False)
        bin_centers_x = (xedges[:-1]+xedges[1:])/2.0
        bin_centers_y = (yedges[:-1]+yedges[1:])/2.0
        x = np.linspace(np.min(bin_centers_x),np.max(bin_centers_x), 1000)
        y = np.linspace(np.min(bin_centers_y),np.max(bin_centers_y), 1000)
        #calculate the cov_matrix
        cov_matrix = np.cov(xs,ys)
        #we need the lambdas
        lamb, vec = np.linalg.eig(cov_matrix)
        #how much are the eigenvectors rotated (hopefully the first is the largest)
        alpha = np.arctan(vec[1][0]/vec[0][0])*180.0/np.pi
        newalpha = np.arctan(vec[1][0]/vec[0][0])
        elsigma = patches.Ellipse(xy=(np.mean(xs),np.mean(ys)),width=2*np.sqrt(2.30*lamb[0]), height=2*np.sqrt(2.30*lamb[1]), angle=alpha, fill=False, color="red")
        el2sigma = patches.Ellipse(xy=(np.mean(xs),np.mean(ys)),width=2*np.sqrt(6.18*lamb[0]), height=2*np.sqrt(6.18*lamb[1]), angle=alpha, fill=False, color="red")
        el3sigma = patches.Ellipse(xy=(np.mean(xs),np.mean(ys)),width=2*np.sqrt(11.83*lamb[0]), height=2*np.sqrt(11.83*lamb[1]), angle=alpha, fill=False, color="red")
        s = plt.gca()
        s.add_artist(elsigma)
        s.add_artist(el2sigma)
        s.add_artist(el3sigma)
        H = H.T
        fixed_x = np.linspace(-0.2,0.2, 12)
        fixed_y = np.linspace(-0.06,0.06,12)
        X, Y = np.meshgrid(bin_centers_x, bin_centers_y)
        plt.pcolormesh(X, Y, H, cmap="PuBu_r")

        #poly = np.polyfit([0,popt[1]*np.cos(popt[3])], [0,-popt[1]*np.sin(popt[3])] , 1)
        poly = np.polyfit([np.mean(xs),np.mean(xs) + vec[0][0]], [np.mean(ys), np.mean(ys) + vec[1][0]],1)
        poly1 = np.polyfit([np.mean(xs), np.mean(xs) + vec[0][1]], [np.mean(ys), np.mean(ys) + vec[1][1]], deg=1)
        xs_new = np.linspace(np.mean(xs)-np.sqrt(11.83*lamb[0])*vec[0][0], np.mean(xs)+np.sqrt(11.83*lamb[0])*vec[0][0], 1000)
        xs_new2 =np.linspace(np.mean(xs)-np.sqrt(11.83*lamb[1])*vec[0][1], np.mean(xs)+np.sqrt(11.83*lamb[1])*vec[0][1], 1000) 
        #if poly[0] < 0:
            #this is probably wrong take the other semi axis
         #   poly = np.polyfit([0, popt[0]*np.sin(popt[3])], [0, popt[0]*np.cos(popt[3])], 1)
        plt.plot(xs_new, xs_new*poly[0] + poly[1], label=str(np.round(poly[0],2)) + "x + " + str(np.round(poly[1],2)))
        plt.plot(xs_new2, xs_new2*poly1[0] + poly1[1], label=str(np.round(poly1[0],2)) + "x + " + str(np.round(poly1[1],2)))
        plt.xlabel("NN error")
        plt.ylabel("MC error")
        plt.title("Error in visible correlation length of NNN")
        plt.legend()
        #plt.xlim(xmin,xmax)
        #plt.ylim(ymin, ymax)
        plt.show()
        #plt.savefig("data/phi4_errors/cl_v_nnn_bj=" + str(betaj) +"bs=" + str(batch_size) + "_cs=2048_auto.png")
        plt.clf()

        H, xedges, yedges = np.histogram2d(ys2,ys,range=limits,bins=60, normed=False)
        bin_centers_x = (xedges[:-1]+xedges[1:])/2.0
        bin_centers_y = (yedges[:-1]+yedges[1:])/2.0
        #calculate the cov_matrix
        cov_matrix = np.cov(ys2,ys)
        #we need the lambdas
        lamb, vec = np.linalg.eig(cov_matrix)
        #how much are the eigenvectors rotated (hopefully the first is the largest)
        alpha = np.arctan(vec[1][0]/vec[0][0])*180.0/np.pi
        newalpha = np.arctan(vec[1][0]/vec[0][0])
        elsigma = patches.Ellipse(xy=(np.mean(ys2),np.mean(ys)),width=2*np.sqrt(2.30*lamb[0]), height=2*np.sqrt(2.30*lamb[1]), angle=alpha, fill=False, color="red")
        el2sigma = patches.Ellipse(xy=(np.mean(ys2),np.mean(ys)),width=2*np.sqrt(6.18*lamb[0]), height=2*np.sqrt(6.18*lamb[1]), angle=alpha, fill=False, color="red")
        el3sigma = patches.Ellipse(xy=(np.mean(ys2),np.mean(ys)),width=2*np.sqrt(11.83*lamb[0]), height=2*np.sqrt(11.83*lamb[1]), angle=alpha, fill=False, color="red")
        s = plt.gca()
        s.add_artist(elsigma)
        s.add_artist(el2sigma)
        s.add_artist(el3sigma)
        x = np.linspace(np.min(bin_centers_x),np.max(bin_centers_x), 1000)
        y = np.linspace(np.min(bin_centers_y),np.max(bin_centers_y), 1000)
       
        H = H.T
        X, Y = np.meshgrid(bin_centers_x, bin_centers_y)
        plt.pcolormesh(X, Y, H, cmap="PuBu_r")
        plt.xlabel("NN error")
        plt.ylabel("MC error")
        #poly = np.polyfit([0,popt[1]*np.cos(popt[3])], [0,-popt[1]*np.sin(popt[3])] , 1)
        poly = np.polyfit([np.mean(ys2),np.mean(ys2) + vec[0][0]], [np.mean(ys), np.mean(ys) + vec[1][0]],1)
        poly1 = np.polyfit([np.mean(ys2), np.mean(ys2) + vec[0][1]], [np.mean(ys), np.mean(ys) + vec[1][1]], deg=1)
        ys_new = np.linspace(np.mean(ys2)-np.sqrt(11.83*lamb[0])*vec[0][0], np.mean(ys2)+np.sqrt(11.83*lamb[0])*vec[0][0], 1000)
        ys_new2 =np.linspace(np.mean(ys2)-np.sqrt(11.83*lamb[1])*vec[0][1], np.mean(ys2)+np.sqrt(11.83*lamb[1])*vec[0][1], 1000) 
        #if poly[0] < 0:
            #this is probably wrong take the other semi axis
         #   poly = np.polyfit([0, popt[0]*np.sin(popt[3])], [0, popt[0]*np.cos(popt[3])], 1)
        plt.plot(ys_new, ys_new*poly[0] + poly[1], label=str(np.round(poly[0],2)) + "x + " + str(np.round(poly[1],2)))
        plt.plot(ys_new2, ys_new2*poly1[0] + poly1[1], label=str(np.round(poly1[0],2)) + "x + " + str(np.round(poly1[1],2)))
        plt.title("Error in hidden correlation legnth of NN")
        plt.legend()
        plt.xlim(xmin,xmax)
        plt.ylim(ymin, ymax)
        #plt.show()
        plt.savefig("data/phi4_errors/cl_h_nn_bj=" + str(betaj) +"bs=" + str(batch_size) + "_cs=2048_auto.png")
        plt.clf()

        #Plot error MC error vs error in network parameter
        xs = []
        ys = []
        c = csv.reader(open("./scripts/data/phi4_fixedpoint/data/error_scatter_" + str(betaj) +"_bs=" + str(batch_size) + "_cs=2048.csv"), delimiter=",")
        for line in c:
            xs += [float(line[0]) / float(betaj)]
            ys += [float(line[1]) / float(betaj)]
        xs = np.array(xs, dtype=float)
        xs = xs
        #xs =xs - np.mean(xs)
        As = np.array(xs)
        AnnError = xs 
        ys = np.array(ys, dtype=float)
        ys = ys
        #ys = ys - np.mean(ys)
        H, xedges, yedges = np.histogram2d(xs,ys, range=limits, bins=60, normed=False)
        bin_centers_x = (xedges[:-1]+xedges[1:])/2.0
        bin_centers_y = (yedges[:-1]+yedges[1:])/2.0
        test = np.max(H[0])/np.max(H[1])
        x = np.linspace(np.min(bin_centers_x),np.max(bin_centers_x), 1000)
        y = np.linspace(np.min(bin_centers_y),np.max(bin_centers_y), 1000)
        cov_matrix = np.cov(xs,ys)
        #we need the lambdas
        if(not np.isfinite(cov_matrix).all()):
            continue
        lamb, vec = np.linalg.eig(cov_matrix)
        #how much are the eigenvectors rotated (hopefully the first is the largest)
        alpha = np.arctan(vec[1][0]/vec[0][0])*180.0/np.pi
        newalpha = np.arctan(vec[1][0]/vec[0][0])
        elsigma = patches.Ellipse(xy=(np.mean(xs),np.mean(ys)),width=2*np.sqrt(2.30*lamb[0]), height=2*np.sqrt(2.30*lamb[1]), angle=alpha, fill=False, color="red")
        el2sigma = patches.Ellipse(xy=(np.mean(xs),np.mean(ys)),width=2*np.sqrt(6.18*lamb[0]), height=2*np.sqrt(6.18*lamb[1]), angle=alpha, fill=False, color="red")
        el3sigma = patches.Ellipse(xy=(np.mean(xs),np.mean(ys)),width=2*np.sqrt(11.83*lamb[0]), height=2*np.sqrt(11.83*lamb[1]), angle=alpha, fill=False, color="red")
        s = plt.gca()
        s.add_artist(elsigma)
        s.add_artist(el2sigma)
        s.add_artist(el3sigma)
        x = np.linspace(np.min(bin_centers_x),np.max(bin_centers_x), 1000)
        y = np.linspace(np.min(bin_centers_y),np.max(bin_centers_y), 1000)
        H = H.T
        X, Y = np.meshgrid(bin_centers_x, bin_centers_y)
        plt.pcolormesh(X, Y, H, cmap="PuBu_r")
        plt.xlabel("NN error")
        plt.ylabel("MC error")
        #poly = np.polyfit([0,popt[1]*np.cos(popt[3])], [0,-popt[1]*np.sin(popt[3])] , 1)
        poly = np.polyfit([np.mean(xs),np.mean(xs) + vec[0][0]], [np.mean(ys), np.mean(ys) + vec[1][0]],1)
        poly1 = np.polyfit([np.mean(xs), np.mean(xs) + vec[0][1]], [np.mean(ys), np.mean(ys) + vec[1][1]], deg=1)
        xs_new = np.linspace(np.mean(xs)-np.sqrt(11.83*lamb[0])*vec[0][0], np.mean(xs)+np.sqrt(11.83*lamb[0])*vec[0][0], 1000)
        xs_new2 =np.linspace(np.mean(xs)-np.sqrt(11.83*lamb[1])*vec[0][1], np.mean(xs)+np.sqrt(11.83*lamb[1])*vec[0][1], 1000) 
        #if poly[0] < 0:
            #this is probably wrong take the other semi axis
         #   poly = np.polyfit([0, popt[0]*np.sin(popt[3])], [0, popt[0]*np.cos(popt[3])], 1)
        plt.plot(xs_new, xs_new*poly[0] + poly[1], label=str(np.round(poly[0],2)) + "x + " + str(np.round(poly[1],2)))
        plt.plot(xs_new2, xs_new2*poly1[0] + poly1[1], label=str(np.round(poly1[0],2)) + "x + " + str(np.round(poly1[1],2)))
        plt.title("error in network parameter")
        plt.legend()
        plt.xlim(xmin,xmax)
        plt.ylim(ymin, ymax)
        #plt.show()
        plt.savefig("data/phi4_errors/network_error_bj=" + str(betaj) +"bs=" + str(batch_size) + "_cs=2048_auto.png")
        plt.clf()
        #plot error in network parameter vs error in correlation length
        poly = np.polyfit(AnnError, corrLenghtError, 1)
        x = np.linspace(np.min(AnnError), np.max(AnnError), 1000)
        plt.plot(x, x *poly[0] + poly[1], label="y="+str(np.round(poly[0], 2))+"x +" + str(np.round(poly[1],2)))
        plt.plot(AnnError, corrLenghtError, "ro")
        plt.xlabel("error in A")
        plt.ylabel("NN error")
        plt.title("Error in A vs error in correlation length (visible)")
        plt.legend()
        plt.xlim(xmin,xmax)
        plt.ylim(ymin, ymax)
        #plt.show()
        plt.savefig("data/phi4_errors/cr_a_v_bj=" + str(betaj) +"bs=" + str(batch_size) + "_cs=2048_auto.png")
        plt.clf()
        #poly, cov = np.polyfit(AnnError, hiddenCorrLengthError, 1, cov=True)

        poly, yerror = lin_fit.lin_fit(AnnError, hiddenCorrLengthError)
        px, confs = lin_fit.conf_calc(AnnError, yerror)
        slopes[np.round(float(betaj),2)] += [poly[0]]
        slope_cov[np.round(float(betaj),2)] += [confs]
        plt.plot(x, x *poly[0] + poly[1], label="y="+str(np.round(poly[0],2))+"x +" + str(np.round(poly[1],2)))
        plt.plot(AnnError, hiddenCorrLengthError, "ro")
        plt.xlabel("error in A")
        plt.ylabel("NN error")
        plt.title("Error in A vs error in correlation length (hidden)")
        plt.legend()
        #plt.show()
        plt.xlim(xmin,xmax)
        plt.ylim(ymin, ymax)
        plt.savefig("data/phi4_errors/cr_a_h_bj=" + str(betaj) +"_bs=" + str(batch_size) + "_cs=2048_auto.png")
        plt.clf()


#plot slope
# parax = np.linspace(0.6,0.9, 1000)
# paray = parax**2
# plt.plot(parax,paray, "y-.")
# dicKeys = list(slopes.keys())
# plt.plot(dicKeys[0], dicKeys[0]**2, "ro", label=r"$\beta J$ = 0.8")
# slopex = np.linspace(-0.05, 0.05,1000)
# plt.plot(dicKeys[0] + slopex,dicKeys[0]**2  +  np.mean(slopes[dicKeys[0]])*slopex, "r-")
# plt.fill_between(dicKeys[0] + slopex, dicKeys[0]**2  +  (np.mean(slopes[dicKeys[0]]))*slopex + np.abs(slope_cov[dicKeys[0]][0]),dicKeys[0]**2  +  (np.mean(slopes[dicKeys[0]]))*slopex - np.abs(slope_cov[dicKeys[0]][0]), color="red",alpha=0.5)

# plt.plot(dicKeys[1],dicKeys[1]**2, "go", label=r"$\beta J$ = 1.0")
# plt.plot(dicKeys[1] + slopex, dicKeys[1]**2 + np.mean(slopes[dicKeys[1]])*(slopex), "g-")
# plt.fill_between(dicKeys[1] + slopex ,dicKeys[1]**2  +  (np.mean(slopes[dicKeys[1]]))*slopex + np.abs(slope_cov[dicKeys[1]][0]),dicKeys[1]**2  +  (np.mean(slopes[dicKeys[1]]))*slopex - np.abs(slope_cov[dicKeys[1]][0]), color="green", alpha=0.5)

# plt.plot(dicKeys[2], dicKeys[2]**2, "bo", label=r"$\beta J$ = 1.2")
# plt.plot(dicKeys[2] + slopex, dicKeys[2]**2 + np.mean(slopes[dicKeys[2]])*slopex, "b-")
# plt.fill_between(dicKeys[2] + slopex,dicKeys[2]**2  +  (np.mean(slopes[dicKeys[2]]))*slopex + np.abs(slope_cov[dicKeys[2]][0]),dicKeys[2]**2  +  (np.mean(slopes[dicKeys[2]]))*slopex - np.abs(slope_cov[dicKeys[2]][0]), color="blue", alpha=0.5)

# plt.xlabel(r"$tanh(\beta J)$")
# plt.ylabel(r"$tanh(\beta J)$")
# plt.legend()
# plt.savefig("data/new_errors/slope_from_error.png")
# plt.clf()


# #closer look at confidence interval
# parax = np.linspace(0.6,0.73, 1000)
# paray = parax**2
# plt.plot(parax,paray, "y-.")
# dicKeys = list(slopes.keys())
# plt.plot(dicKeys[0], dicKeys[0]**2, "ro", label=r"$\beta J$ = 0.8")
# slopex = np.linspace(-0.05, 0.05,1000)
# plt.plot(dicKeys[0] + slopex,dicKeys[0]**2  +  np.mean(slopes[dicKeys[0]])*slopex, "r-")
# plt.fill_between(dicKeys[0] + slopex, dicKeys[0]**2  +  (np.mean(slopes[dicKeys[0]]))*slopex + np.abs(slope_cov[dicKeys[0]][0]),dicKeys[0]**2  +  (np.mean(slopes[dicKeys[0]]))*slopex - np.abs(slope_cov[dicKeys[0]][0]), color="red",alpha=0.5)
# plt.xlabel(r"$tanh(\beta J)$")
# plt.ylabel(r"$tanh(\beta J)$")
# plt.legend()
# plt.savefig("data/new_errors/slope_from_error_bj=0_8.png")
# plt.clf()

# parax = np.linspace(0.7,0.83, 1000)
# paray = parax**2
# plt.plot(parax,paray, "y-.")
# dicKeys = list(slopes.keys())
# plt.plot(dicKeys[1], dicKeys[1]**2, "go", label=r"$\beta J$ = 1.0")
# slopex = np.linspace(-0.05, 0.05,1000)
# plt.plot(dicKeys[1] + slopex,dicKeys[1]**2  +  np.mean(slopes[dicKeys[1]])*slopex, "g-")
# plt.fill_between(dicKeys[1] + slopex, dicKeys[1]**2  +  (np.mean(slopes[dicKeys[1]]))*slopex + np.abs(slope_cov[dicKeys[1]][0]),dicKeys[1]**2  +  (np.mean(slopes[dicKeys[1]]))*slopex - np.abs(slope_cov[dicKeys[1]][0]), color="green",alpha=0.5)
# plt.xlabel(r"$tanh(\beta J)$")
# plt.ylabel(r"$tanh(\beta J)$")
# plt.legend()
# plt.savefig("data/new_errors/slope_from_error_bj=1_0.png")
# plt.clf()

# parax = np.linspace(0.77,0.9, 1000)
# paray = parax**2
# plt.plot(parax,paray, "y-.")
# dicKeys = list(slopes.keys())
# plt.plot(dicKeys[2], dicKeys[2]**2, "bo", label=r"$\beta J$ = 1.0")
# slopex = np.linspace(-0.05, 0.05,1000)
# plt.plot(dicKeys[2] + slopex,dicKeys[2]**2  +  np.mean(slopes[dicKeys[2]])*slopex, "b-")
# plt.fill_between(dicKeys[2] + slopex, dicKeys[2]**2  +  (np.mean(slopes[dicKeys[2]]))*slopex + np.abs(slope_cov[dicKeys[2]][0]),dicKeys[2]**2  +  (np.mean(slopes[dicKeys[2]]))*slopex - np.abs(slope_cov[dicKeys[2]][0]), color="blue",alpha=0.5)
# plt.xlabel(r"$tanh(\beta J)$")
# plt.ylabel(r"$tanh(\beta J)$")
# plt.legend()
# plt.savefig("data/new_errors/slope_from_error_bj=1_2.png")
# plt.clf()