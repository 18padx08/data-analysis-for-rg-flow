import numpy as np
import matplotlib.pyplot as plt

def transformV2H(i,j):
    return (1.0/4.0*(2*(i-j -1) + 32),1.0/4.0*(2*(i+j +1) - 32))
def transformH2V(i,j):
    return (i+j, (j-i) + (15))
i_s = range(0,32)
j_s = range(0,32)
for i in i_s:
    for j in j_s:
        if (i%2 ==0 and j%2 ==0) or ((i+j)%2 == 0):
            plt.plot(i,j,"ro")
        else:
            plt.plot(i,j,"b*")

tmp1 = transformV2H(0,1)
tmp2 = transformV2H(1,0)
tmp3 = transformV2H(2,1)
tmp4 = transformV2H(1,2)
print(tmp1,tmp2,tmp3,tmp4)

p1 = np.array((0,0))
p2 = np.array((p1[0] +1, p1[1] + 0))
p3 = np.array((p1[0] +1,p1[0] + 1))
p4 = np.array((p1[0] + 0, p1[0] + 1))

p1 = transformH2V(*tmp1)
p2 = transformH2V(*tmp2)
p3 = transformH2V(*tmp3)
p4 = transformH2V(*tmp4)

p1 = (p1[0],p1[1])
p2 = (p2[0],p2[1])
p3 = (p3[0],p3[1])
p4 = (p4[0],p4[1])
plt.xlabel("i")
plt.ylabel("j")
plt.plot(*p1,"g*")
plt.plot(*p2,"g*")
plt.plot(*p3,"g*")
plt.plot(*p4,"g*")


plt.show()