import numpy as np
from matplotlib import pyplot as plt

a = 1
c = -3
l = np.pi
N = 10
K = 1000
T = 1
h = l/N
tau = T/K
sigma = (a**2 * tau**2) / h**2

tk = np.zeros(K)
xn = np.zeros(N)

for i in range(N):
    xn[i] = h * i
for i in range(K):
    tk[i] = tau * i

#Дано

def phi0(t):
    return np.sin(2*t)

def phi1(t):
    return np.sin(2*t) * (-1)

def psi1():
    return 0

def psi2(x):
    return 2 * np.cos(x)

def exact(x,t):
    return np.cos(x) * np.sin(2*t)

# Явная схема

def explicit():

    u = np.zeros((K,N))

    for k in range(0,K):
        u[k][0] = phi0(tau * k)
        u[k][-1] = phi1(tau * k)

    for j in range(0,N):
        u[0][j] = psi1()
        u[1][j] = psi2(j * h) * tau

    for k in range(1,K-1):
        for j in range(1,N-1):
            u[k+1][j] = u[k][j + 1] * (a ** 2 * tau ** 2) / h ** 2 + u[k][j] * (-2 * (a ** 2 * tau ** 2) / h ** 2 + 2 + c * tau ** 2) + u[k][j - 1] * (a ** 2 * tau ** 2) / h ** 2 - u[k - 1][j]

        u[k][0] = phi0(k * tau)
        u[k][-1] = phi1(k * tau)

    return u

# Неявная схема

def implicit():

    u = np.zeros((K,N))

    for k in range(0,K):
        u[k][0] = phi0(tau * k)
        u[k][-1] = phi1(tau * k)

    for j in range(0,N):
        u[0][j] = psi1()
        u[1][j] = psi2(j * h) * tau

    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)

    for k in range(1,K-1):

        a[0] = 0
        b[0] = 1
        c[0] = 0
        d[0] = phi0(tau * k)

        for j in range(1,N-1):
            a[j] = sigma
            b[j] = 2 * (1 - sigma)
            c[j] = sigma
            d[j] = u[k+1][j] + u[k-1][j]

        a[-1] = 0
        b[-1] = 1
        c[-1] = 0
        d[-1] = phi1(tau * k)

        u[k] = swp(a,b,c,d)

    return u

# Прогонка

def swp(a,b,c,d):

    i = np.shape(d)[0]

    def searchP():
        P = np.zeros(i)
        P[0] = -c[0] / b[0]
        for j in range(1, len(P)):
            P[j] = -c[j] / (b[j] + a[j] * P[j - 1])
        return P

    def searchQ():
        Q = np.zeros(i)
        Q[0] = d[0] / b[0]
        for j in range(1, len(Q)):
            Q[j] = (d[j] - a[j] * Q[j - 1]) / (b[j] + a[j] * P[j - 1])
        return Q

    def searchX():
        X = np.zeros(i)
        X[i - 1] = Q[i - 1]
        for j in range(len(X) - 2, -1, -1):
            X[j] = P[j] * X[j + 1] + Q[j]
        return X

    P = searchP()
    Q = searchQ()
    X = searchX()

    return X

# Погрешность

def accuracy_error(tk,xn,u):
    res = np.zeros(K)
    for i in range(K):
        res[i] = np.max(np.abs(u[i] - np.array([exact(x,tk[i]) for x in xn])))

    plt.figure(figsize=(16, 8))
    plt.plot(tk[1:], res[1:])
    plt.xlabel('t')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()

# Main

u1 = explicit()
u2 = implicit()

accuracy_error(tk,xn,u1)
accuracy_error(tk,xn,u2)