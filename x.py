from vectors import Point, Vector
import numpy as np

matrix_size = 4
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([4.0, 5.0, 6.0])
v3 = np.array([7.0, 8.0, 9.0])
v4 = np.array([1.0, 2.0, 3.0])
S = [v1, v2, v3, v4]

A = [np.array([1, 2, 3, 5]),
     np.array([4, 5, 4, 4]),
     np.array([7, 8, 9, 4]),
     np.array([10, 11, 12, 13])]

B = [np.array([1.0, 2.0, 3.0]),
     np.array([4.0,5.0, 6.0]),
     np.array([4.0, 5.0, 6.0]),
     np.array([1.0, 2.0, 11.0])]

def AS():
    Prod = [0] * matrix_size
    for i in range(0, matrix_size):
        Prod[i] = np.dot(A[i], S)
    return Prod


def grad():
    tmp = AS()
    for i in range(0, matrix_size):
        tmp[i] = tmp[i] + B[i]
    return tmp

def E(S):
    AS = [0] * matrix_size
    for i in range(0, matrix_size):
        AS[i] = np.dot(A[i], S)
    #print(AS)
    sua = 0
    for i in range(0, matrix_size):
        sua = sua + np.dot(AS[i], S[i])
        #print(np.dot(AS[i], S[i]))
    sub = 0
    for i in range(0, matrix_size):
        sub = sub + np.dot(S[i], B[i])
    res = sua + sub
    #print(res)
    return res

def normalize():
    for i in range(0, matrix_size):
        t = S[i][0] * S[i][0] + S[i][1] * S[i][1] + S[i][2] * S[i][2]
        t =  np.sqrt(t)
        S[i][0] = S[i][0] / t
        S[i][1] = S[i][1] / t
        S[i][2] = S[i][2] / t
    return S

print(grad)



for i in range(0, 1000):
    gradient = grad()
    for j in range(0, matrix_size):
        S[j] = S[j] - 0.001 * gradient[j]
    S = normalize()
    print(E(S))
print(E(S))




# v4 = np.dot(v1, v2) -- скалярное произведение векторов





