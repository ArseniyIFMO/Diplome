from time import process_time

import numpy as np

# matrix_size = 16


SX = 6
SY = 6
x = np.array([1.0, 0.0, 0.0])
y = np.array([0.0, 1.0, 0.0])
z = np.array([0.0, 0.0, 1.0])
S = np.random.randn(SX + 2, SY + 2, 3)
for i in range(0, SX + 2):
    S[i][0] = np.array([0, 0, 0])
    S[i][SY + 1] = np.array([0, 0, 0])

for i in range(0, SY + 2):
    S[0][i] = np.array([0, 0, 0])
    S[SX + 1][i] = np.array([0, 0, 0])

K = 0.4
D = 0.35
J = 1
step = 0.1

# print(S.shape)
# np.roll
# np.cross
# alpha
# steps = 1000
def normalPrintS():
    for i in range(0, SX + 2):
        print()
        for j in range(0, SY + 2):
            print(S[i][j], end=" ")
    print()


def grad(i, j):
    tmp = J * (S[i + 1][j] + S[i - 1][j] + S[i][j + 1] + S[i][j - 1])
    tmp2 = D * (np.cross(S[i + 1][j], x) + np.cross(S[i][j + 1], y)
                - np.cross(S[i - 1][j], x) - np.cross(S[i][j - 1], y))
    tmp3 = 2 * z * K * np.dot(z, S[i][j]).item()
    res = - tmp + tmp2 - tmp3
    return res


def E():
    res = 0
    for i in range(1, SX + 1):
        for j in range(1, SY + 1):
            tmp = J * np.dot((S[i + 1][j] + S[i - 1][j]
                              + S[i][j + 1] + S[i][j - 1]), S[i][j])
            tmp2 = D * (+ np.dot(np.cross(S[i + 1][j], S[i][j]), x)
                        + np.dot(np.cross(S[i][j + 1], S[i][j]), y)
                        - np.dot(np.cross(S[i - 1][j], S[i][j]), x)
                        - np.dot(np.cross(S[i][j - 1], S[i][j]), y))
            tmp3 = K * (np.dot(z, S[i][j]) ** 2)
            res = res - tmp / 2 - tmp2 / 2 - tmp3
    return res

def E2():
    res = 0
    for i in range(1, SX + 1):
        for j in range(1, SY + 1):
            res += np.dot(S[i][j], grad(i,j))
    return res * 0.5

def normalize():
    for i in range(1, SX + 1):
        for j in range(1, SY + 1):
            norm = (
            S[i][j][0] * S[i][j][0] +
            S[i][j][1] * S[i][j][1] +
            S[i][j][2] * S[i][j][2])
            norm = np.sqrt(norm)
            S[i][j][0] = S[i][j][0] / norm
            S[i][j][1] = S[i][j][1] / norm
            S[i][j][2] = S[i][j][2] / norm

normalize()

prev_grad =  np.random.randn(SX + 2, SY + 2, 3)
cur_grad = np.random.randn(SX + 2, SY + 2, 3)
for i in range(1, SX + 1):
    for j in range(1, SY + 1):
        prev_grad[i][j] = grad(i, j)
omega = 1
k = 0
maxNorm =  100
t1 = process_time()
while(maxNorm > 0.001):
    newS = np.zeros_like(S)
    maxNorm = 0
    sum_of_prev_grad = 0
    sum_of_grad  =  0
    for i in range(1, SX + 1):
        for j in range(1, SY + 1):
            cur_grad[i][j] = grad(i,  j)

    for i in range(1, SX + 1):
        for j in range(1, SY + 1):
            sum_of_prev_grad =  sum_of_prev_grad + (
                np.dot(prev_grad[i][j], prev_grad[i][j]))

    for i in range(1, SX + 1):
        for j in range(1, SY + 1):
            sum_of_grad = sum_of_grad + (
                np.dot(cur_grad[i][j], cur_grad[i][j]))

        omega = sum_of_grad / sum_of_prev_grad

    for i in range(1, SX + 1):
        for j in range(1, SY + 1):
            g = grad(i,  j)
            projGradOnS = np.dot(S[i][j], g)
            g = g - projGradOnS * S[i][j]
            maxNorm = np.maximum(maxNorm, np.linalg.norm(g))
            if(k == 0):
                prev_grad[i][j] = grad(i, j)
                newS[i][j] = S[i][j] - step * (grad(i, j))
            else:
                newS[i][j] = (S[i][j] - step * (grad(i, j)) -
                              ( step * omega *  prev_grad[i][j] ))
                prev_grad[i][j] = (
                    grad(i, j) + prev_grad[i][j] * omega)
    S = newS
    normalize()
    #print(E(),maxNorm)
    k = k + 1
t2 = process_time()
#print(E(), maxNorm)
print(k)
print(t2 - t1)



# v4 = np.dot(v1, v2) -- скалярное произведение векторов


#
# v1 = np.array([1.0, 2.0, 3.0])
# v2 = np.array([4.0, 5.0, 6.0])
# v3 = np.array([7.0, 8.0, 9.0])
# v4 = np.array([1.0, 2.0, 3.0])
# S = [v1, v2, v3, v4]
#
# A = [np.array([1, 2, 3, 5]),
#      np.array([4, 5, 4, 4]),
#      np.array([7, 8, 9, 4]),
#      np.array([10, 11, 12, 13])]
#
# B = [np.array([1.0, 2.0, 3.0]),
#      np.array([4.0,5.0, 6.0]),
#      np.array([4.0, 5.0, 6.0]),
#      np.array([1.0, 2.0, 11.0])]
#
# def AS():
#     Prod = np.zeros(matrix_size, dtype = np.float64)
#     for i in range(0, matrix_size):
#         Prod[i] = np.dot(A[i], S)
#     return Prod
#
#
# def grad():
#     tmp = AS()
#     for i in range(0, matrix_size):
#         tmp[i] = tmp[i] + B[i]
#     return tmp
#
# def E(S):
#     AS = [0] * matrix_size
#     for i in range(0, matrix_size):
#         AS[i] = np.dot(A[i], S)
#     #print(AS)
#     sua = 0
#     for i in range(0, matrix_size):
#         sua = sua + np.dot(AS[i], S[i])
#         #print(np.dot(AS[i], S[i]))
#     sub = 0
#     for i in range(0, matrix_size):
#         sub = sub + np.dot(S[i], B[i])
#     res = sua + sub
#     #print(res)
#     return res
#
# def normalize():
#     for i in range(0, matrix_size):
#         t = S[i][0] * S[i][0] + S[i][1] * S[i][1] + S[i][2] * S[i][2]
#         t =  np.sqrt(t)
#         S[i][0] = S[i][0] / t
#         S[i][1] = S[i][1] / t
#         S[i][2] = S[i][2] / t
#     return S
#
# print(grad)
#
#
#
# for i in range(0, 10000):
#     gradient = grad()
#     S = S - 0.001 * gradient
#     print(E(S))
# print(E(S))
#



# v4 = np.dot(v1, v2) -- скалярное произведение векторов





