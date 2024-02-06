import numpy as np
def Manhattendist():
    F=[]
    S=[]
    print("enter the noof vectors")
    a=int(input())
    for i in range(a):
        b=int(input())
        F.append(b)

    H=np.array(F)
    print("enter the second vector")

    for i in range(a):
        c=int(input())
        S.append(c)

    N=np.array(S)
    Z=H-N
    print(Z)
    L=len(Z)

    for i in range (L):
        if(Z[i] < 0):
            Z[i]=-1*Z[i]
    print(Z)
def Euclidian():

    l1=[]
    l2=[]
    print("enter noof vectors")
    a1=int(input())
    for i in range(a1):
        b1=int(input())
        l1.append(b1)

    L1=np.array(l1)
    print("enter the second vector")
    for i in range(a1):
        c1=int(input())
        l2.append(c1)

    L2=np.array(l2)

    z2=L1-L2
    for i in range(a1):
        if (z2[i] ):
            z2[i]=z2[i]*z2[i]
    k=np.sqrt(z2)
    print(k)
Euclidian()
Manhattendist()

#question 2
import numpy as np

def kNN_classifier(X_train, y_train, X_test, k):
    distances = np.linalg.norm(X_train - X_test, axis=1)
    nearest_labels = y_train[np.argsort(distances)[:k]]
    return np.argmax(np.bincount(nearest_labels))


X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([2.5, 3.5])
k_value = 3

predicted_class = kNN_classifier(X_train, y_train, X_test, k_value)
print(f"The predicted class for the test instance is: {predicted_class}")