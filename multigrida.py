import numpy as np
def norm(R):
    max = np.amax(R)
    min = np.amin(R)
    min = min * (-1)
    if(max > min):
        return max
    else:
        return min

def reshape(x):
    sz = x.size
    #print(sz)
    f = np.zeros(2 * sz - 1)
    for i in range(0, sz):
        f[2 * i] = x[i]
    #print(f)
    for i in range(0, sz - 1):
        f[2 * i + 1] = (f[2 * i] + f[2 * i + 2]) / 2
    return f

def makeAWithSize(n):
    myA = np.random.randn(n, n)
    myA = np.zeros_like(myA)
    myA[0][1] = 1
    myA[0][0] = -2
    myA[n - 1][n - 2] = 1
    myA[n - 1][n - 1] = -2
    for i in range(1, n - 1):
        myA[i][i] = -2
        myA[i][i - 1] = 1
        myA[i][i + 1] = 1
    return myA

N = 5
eps = 0.001
n = 9
alpha = 0.01
f = np.zeros(n)
b = np.random.randn(n)
for i in range(0, n):
    b[i] = 1
    f[i] = 1

A = np.random.randn(n, n)
A = np.zeros_like(A)
A[0][1] = 1
A[0][0] = -2
A[n-1][n-2] = 1
A[n-1][n-1] = -2
for i in range(1,  n-1):
    A[i][i] = -2
    A[i][i - 1] = 1
    A[i][i + 1] = 1
for i in range(0, n):
    b[i] = 1


while(1):
    RF = np.copy(b)
    #print(f)
    Af = A.dot(f)
    RF = RF - Af
    f = f - RF * alpha
    if(norm(RF) < eps):
        break

size = 2 * n - 1

f = reshape(f)
print(f)
A = makeAWithSize(size)
print(A)
b = np.random.randn(2 * n - 1)
while(1):
    RF = np.copy(b)
    print(f)
    Af = A.dot(f)
    RF = RF - Af
    f = f - RF * alpha
    if (norm(RF) < eps):
        break