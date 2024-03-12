import matplotlib.pyplot as pl
import numpy as np

c1=[(4,2),(4,2.5),(4.70,3),(4.9,4),(5,4.5),(4.3,3.5),(5.5,3.8)]
c2=[(5,4.8),(7.45,6.8),(8.2,6.8),(7.55,6.89),(6.78,8.5),(7.9,8),(7.89,6)]
Test=(6,6)
Dist1=[]
Dist2=[]

def Euclidian(m,Test):
    sub=m-Test
    M=sub*sub
    sum=0
    for i in range(len(M)):
        sum=sum+M[i]
    Q=np.sqrt(sum)
    return Q

for i in range (len(c1)):
    Z=np.array(Test)
    np.array(c1[i])
    a=c1[i]
    Dist1.append(Euclidian(a,Z))
for i in range (len(c2)):
    Z=np.array(Test)
    np.array(c2[i])
    a=c2[i]
    Dist2.append(Euclidian(a,Z))
sum1=0
sum2=0
for i in range(len(Dist1)):
    sum1=sum1+Dist1[i]
for i in range(len(Dist2)):
    sum2=sum2+Dist2[i]

avg1=sum1/len(Dist1)
avg2=sum2/len(Dist2)

if avg1 < avg2 :
    print("the new test point belongs to : category one")
else:
    print("the new test point belongs to : category two")

Xcordc1=[]
Ycordc1=[]
for i in range(len(c1)):
    K=c1[i]
    for j in range(1):
        Xcordc1.append(K[j])
        Ycordc1.append(K[j+1])

Xc1=np.array(Xcordc1)
Yc1=np.array(Ycordc1)

Xcordc2=[]
Ycordc2=[]

for i in range(len(c2)):
    K=c2[i]
    for j in range(1):
        Xcordc2.append(K[j])
        Ycordc2.append(K[j+1])

Xc2=np.array(Xcordc2)
Yc2=np.array(Ycordc2)

Xtest=Test[0]
YTest=Test[1]

pl.scatter(Xc1,Yc1)
pl.scatter(Xc2,Yc2)
pl.scatter(Xtest,YTest)
pl.show()
