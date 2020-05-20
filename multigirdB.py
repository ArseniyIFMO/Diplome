from time import  process_time

import numpy as np
def norm(R):
    #return np.linalg.norm(R) / (R.shape[0])
    return np.max(np.abs(R))


def reshape(x):
    sz = x.size
    f = np.zeros(2 * sz + 1)
    for i in range(0, sz):
        f[2 * i + 1] = x[i]
    for i in range(1, sz ):
        f[2 * i] = (f[2 * i- 1] + f[2 * i + 1]) / 2
    f[0] = x[0] / 2
    f[-1] = x[-1] / 2
    return f

def makeAWithSize(n):
    myA = np.zeros((n,n), dtype= float)
    myA[0][1] = 1
    myA[0][0] = -2
    myA[n - 1][n - 2] = 1
    myA[n - 1][n - 1] = -2
    for i in range(1, n - 1):
        myA[i][i] = -2
        myA[i][i - 1] = 1
        myA[i][i + 1] = 1
    myA = myA * (n * n)
    return myA

def Af(f):
    r = -2*f
    r[:-1] += f[1:]
    r[1:] += f[:-1]
    #print(f.shape[0])
    return r * ((f.shape[0] + 1) ** 2)



def solveJacobi( b, f, eps = None, alpha = None, maxIter = 1000000):
    #print("In jacobi", f[-5:])
    #print("Norm", norm(f))
    f0 = f
    for n in range(maxIter):
        RF = b - Af(f)
        nrm = norm(RF)
        #if(n == 0):
            #print("Initial residual" , nrm)
            #print("Initial residue ",  RF[0:5])
        if nrm < eps:
            print("Converged for: ", n, nrm, f.shape[0])
            #print("Max" , np.max((f-f0)))
            return f
        #print(nrm, end="\n")
        #print("progress", f[-5:])
        f = f  - RF * alpha

    print("Not converged: ", n, nrm)
    return f

def makeBWithSize(n):
    return np.ones(n)



def unit():
    x = np.random.randn(10)
    A = makeAWithSize(x.shape[0])
    y = A @ x
    z = Af(x)
    YZ = y - z
    assert(np.linalg.norm(YZ) < 0.001)




N = 5
eps = 1e-5
alpha = 0.3


t1 = process_time()

f = np.zeros(N, dtype=float)

for i in range(7):
    N = f.shape[0]
    #A = makeAWithSize(N)
    B = makeBWithSize(N)

    f = solveJacobi( B, f, eps = eps, alpha = alpha / (N ** 2))
    #print("Solution    ", f[0:5])
    #print("Solution    ", f[-5:])
    f = reshape(f)
    #print("Interpolated", f[0:5])
    #print("Interpolated", f[-5:])
t2 = process_time()

f = np.zeros(N, dtype=float)
N = f.shape[0]
A = makeAWithSize(N)
B = makeBWithSize(N)
f = solveJacobi( B, f, eps=eps, alpha=alpha / (N ** 2))

print(t2 - t1)