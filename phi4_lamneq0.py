import numpy as np
import matplotlib.pyplot as plt


AGauss= 1
def func(x,lamb, mean,var):
    factor = 1.0/(2.0*var**2)
    return np.exp(- factor*(x - mean)**2 - lamb * (x**2 - 1)**2)

def gauss(x, mean, var):
    return AGauss* np.exp( - (1.0/(2.0 *var**2))*(x- mean)**2)
xs = np.linspace(-4,4,1000)

for l in [0,5,10]:
    for m in [-2]:
        ys = []
        lamb = l
        mean = m
        meanGauss = m
        var = 1.0/np.sqrt(2)
        varGauss = 1.0/np.sqrt(2)

        #calculate discriminant
        if lamb > 0:
            s = (2-4*lamb)/(4*lamb)
            t = (-2*mean)/(4*lamb)
            p = s
            q = t
            D = (p/3)**3 + (q/2)**2
        else:
            D=0
        print(D)
        #print(p,q)
        x_zero = 0
        if D>0:
            uCubed = -(q/2) + np.sqrt(D)
            vCubed = -(q/2) - np.sqrt(D)
            u =0
            v=0
            if uCubed < 0:
                u = - abs(uCubed)**(1.0/3)
            else:
                u = uCubed**(1.0/3)
            if vCubed < 0:
                v = - abs(vCubed)**(1.0/3)
            else:
                v = vCubed**(1.0/3)
            x_zero =  u+v
            print(x_zero)
            meanGauss = x_zero
            AGauss = func(x_zero,lamb,mean,var)/gauss(x_zero,meanGauss,varGauss)
        elif D<0:
            #we have three real solutions
            rho = np.sqrt(-p**3/27)
            theta = np.arccos(-q/(2*rho))
            y1 = 2*(rho**(1.0/3.0))*np.cos(theta/3)
            y2 = 2*(rho**(1.0/3.0))*np.cos(theta/3 + 2*3.14159/3)
            y3 = 2*(rho**(1.0/3.0))*np.cos(theta/3 + 4*3.14159/3)
            ally = np.array([y1,y2,y3])
            #print(ally)
            maxY = np.argmax(func(ally,lamb,mean,var))
            meanGauss = ally[maxY]
            #print(func(meanGauss,lamb,mean,var),gauss(meanGauss,meanGauss,var))
            AGauss = func(meanGauss,lamb,mean,var)
            #double variance until everything is under the envelope
            check = func(ally,lamb,mean,var)/gauss(ally,meanGauss,varGauss)
            print(check)
            check = check[check > 1]
            print(check)
            while len(check) >0:
                varGauss *= 2
                check = func(ally,lamb,mean,var)/gauss(ally,meanGauss,varGauss)
                print(check)
                check = check[check > 1]
            print(AGauss)
        while len(ys) < 10000:
            tmpgaus = np.random.normal(meanGauss, varGauss)
            prop = np.min((1.0,func(tmpgaus,lamb,mean,var)/gauss(tmpgaus,meanGauss,varGauss)))
            p = np.random.uniform(0,1,1)
            if p < prop:
                ys += [tmpgaus]
        print(np.mean(ys), np.var(ys))
        alotofrandom = np.random.normal(meanGauss, varGauss,10000)
        hi,box = np.histogram(alotofrandom, bins=100,range=[-4,4],normed=True)
        hi = hi*2
        box = (box[1:] + box[:-1])/2.0
        #plt.plot(box,hi)

        bins = np.linspace(-4,4,)
        ys = np.array(ys)
        hist, boxes = np.histogram(ys,bins=100, range=[-4,4])
        boxes = (boxes[1:] + boxes[:-1])/2.0

        blub = np.max(func(xs,lamb,mean, 1.0/np.sqrt(2)))/np.max(hist)
        hist = hist * blub
        print(np.sum(func(xs,lamb,mean, 1.0/np.sqrt(2))))
        plt.plot(boxes, hist, label="sampled", linewidth=3)
        plt.plot(xs,func(xs,lamb,mean,1.0/np.sqrt(2)), label="P(x)", linestyle="dotted", linewidth=3)
        plt.plot(xs, gauss(xs, meanGauss, varGauss), label=r"$\mathcal{N}$", linestyle="dotted", linewidth=3)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$P(x)$")
        plt.title(r"$\lambda="+str(lamb)+ r"\quad \bar{x}=" + str(mean) +r"$")
        plt.legend()
        plt.savefig("./data/phi4/prob_dist_lamb=" + str(lamb) + "_mean="+str(mean)+".png")
        plt.clf()
        #plt.show()
        #xs = np.linspace(-10,10,1000)
        #ys = 4*lamb*xs**3 + (2-4*lamb)*xs - 2*mean
        #plt.plot(xs,ys)
        #plt.ylim(-0.1,0.1)
        #plt.show()
