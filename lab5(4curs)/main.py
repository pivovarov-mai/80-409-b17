import numpy as np
from matplotlib import pyplot as plt

a = 1
l = np.pi
N = 10
K = 1000
T = 1
h = l/N
tau = T/K
sigma = (a * tau) / h**2

ti = np.zeros(K)
xi = np.zeros(N)

for i in range(N):
    xi[i] = h * i
for i in range(K):
    ti[i] = tau * i

#Дано

def phi0(t):
    return np.exp(-a*t)

def phi1(t):
    return (np.exp(-a*t) * (-1))

def psi(x):
    return np.cos(x)

def analytical_solution(x,t):
    return (np.exp(-a*t)*np.cos(x))

# Явная конечно-разностная схема

def explicit():

    u = np.zeros((K,N))

    for k in range(0,K):
        u[k][0] = phi0(tau * k)

    for j in range(0,N):
        u[0][j] = psi(j * h)

    for k in range(1,K):
        for j in range(1,N-1):
            u[k][j] = sigma * u[k-1][j+1] + (1-2*sigma) * u[k-1][j] + sigma * u[k-1][j-1]

        u[k][0] = phi0(k * tau)
        u[k][-1] = phi1(k * tau)

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

# Неявная конечно-разностная схема

def implicit():

    u = np.zeros((K,N))

    for k in range(0,K):
        u[k][0] = phi0(tau * k)

    for j in range(0,N):
        u[0][j] = psi(j * h)

    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)

    for k in range(1,K):

        a[0] = 0
        b[0] = 1
        c[0] = 0
        d[0] = phi0(tau * k)

        for j in range(1,N-1):
            a[j] = sigma
            b[j] = -(1 + 2 * sigma)
            c[j] = sigma
            d[j] = -u[k-1][j]

        a[-1] = 0
        b[-1] = 1
        c[-1] = 0
        d[-1] = phi1(tau * k)

        u[k] = swp(a,b,c,d)

    return u

# Неявно-явная схема

def implicit_explicit(omega):

    u = np.zeros((K,N))

    imp = implicit()
    exp = explicit()

    for k in range(0, K):
        for j in range(0, N):
            u[k][j] = imp[k][j] * omega + exp[k][j] * (1 - omega)

    return u

# Погрешность

def accuracy_error(ti,xi,u):
    res = np.zeros(K)
    for i in range(K):
        res[i] = np.max(np.abs(u[i] - np.array([analytical_solution(x,ti[i]) for x in xi])))

    plt.figure(figsize=(16, 8))
    plt.plot(ti[1:], res[1:])
    plt.xlabel('t')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()

# Main

u1 = explicit()
u2 = implicit()
u12 = implicit_explicit(0.5)

accuracy_error(ti,xi,u1)
accuracy_error(ti,xi,u2)
accuracy_error(ti,xi,u12)
