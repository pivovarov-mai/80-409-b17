import numpy as np
from matplotlib import pyplot as plt, cm
import math as mt

# Параметры
a = 1
l1 = np.pi/4
l2 = mt.log(2)
l3 = 1
nx = 10
ny = 10
nt = 100
h1 = l1/nx
h2 = l2/ny
tau = l3/nt

#Дано
def phi1(y,t):
    return np.cosh(y)*np.exp(-3*a*t)
def phi2(y,t):
    return 0
def phi3(x,t):
    return np.cos(2*x)*np.exp(-3*a*t)
def phi4(x,t):
    return 5/4*np.cos(2*x)*np.exp(-3*a*t)
def psi(x,y):
    return np.cos(2*x)*np.cosh(y)
def exact(x,y,t):
    return np.cos(2*x)*np.cosh(y)*np.exp(-3*a*t)

# x,y,t
x = np.arange(0, l1+h1, h1)
y = np.arange(0, l2+h2, h2)
t = np.arange(0, l3+tau, tau)

# Аналитическое решение
def analytic(x, y, t):
    ext = np.zeros((len(x), len(y), len(t)))
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            for k in range(0, len(t)):
                ext[i,j,k] = exact(x[i], y[j], t[k])
    return ext

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

# Метод переменных направлений и Метод дробных шагов
def MPN_MDS(f):
    u = np.zeros((len(x), len(y), len(t)))
    for i in range(len(x)):
        for j in range(len(y)):
            u[i,j,0] = psi(x[i],y[j])
    ap = np.zeros(len(x))
    bp = np.zeros(len(x))
    cp = np.zeros(len(x))
    dp = np.zeros(len(x))
    for k in range(1, len(t)):
        u1 = np.zeros((len(x), len(y)))
        th = t[k-1] + tau/2
        v = u[:,:,k-1]
        for j in range(len(y)-1):
            ap[0] = 0
            bp[0] = 1
            cp[0] = 0
            dp[0] = phi1(y[j], th)
            for i in range(1,len(x)-1):
                if (f==1):
                    ap[i] = -(a*tau) / (2*h1*h1)
                    bp[i] = (a*tau) / (h1*h1) + 1
                    cp[i] = -(a*tau) / (2*h1*h1)
                    dp[i] = (a*tau) / (2*h2*h2) * v[i][j-1] + v[i][j] * (1 - (a*tau) / (h2*h2)) + (a*tau) / (2*h2*h2) * v[i][j+1]
                if (f==2):
                    ap[i] = -(a * tau) / (h1 * h1)
                    bp[i] = (2 * a * tau) / (h1 * h1) + 1
                    cp[i] = -(a * tau) / (h1 * h1)
                    dp[i] = v[i][j]
            ap[-1] = 0
            bp[-1] = 1
            cp[-1] = 0
            dp[-1] = phi2(y[j], th)
            prg = swp(ap,bp,cp,dp)
            for i in range(len(x)):
                u1[i,j] = prg[i]
                u1[i,0] = phi3(x[i],th)
                u1[i,-1] = phi4(x[i],th)
        for j in range(len(y)):
            u1[0,j] = phi1(y[j],th)
            u1[-1, j] = phi2(y[j], th)

        u2 = np.zeros((len(x), len(y)))
        for i in range(len(x)-1):
            ap[0] = 0
            bp[0] = 1
            cp[0] = 0
            dp[0] = phi3(x[i], t[k])
            for j in range(1,len(y)-1):
                if (f==1):
                    ap[j] = -(a*tau) / (2*h2*h2)
                    bp[j] = 1 + (a*tau) / (h2*h2)
                    cp[j] = -(a*tau) / (2*h2*h2)
                    dp[j] = (a*tau) / (2*h1*h1) * u1[i-1][j] + u1[i][j] * (1 - (a*tau) / (h1*h1)) + (a*tau) / (2*h1*h1) * u1[i+1][j]
                if (f==2):
                    ap[j] = -(a * tau) / (h2 * h2)
                    bp[j] = 1 + (2 * a * tau) / (h2 * h2)
                    cp[j] = -(a * tau) / (h2 * h2)
                    dp[j] = u1[i][j]
            ap[-1] = 0
            bp[-1] = 1
            cp[-1] = 0
            dp[-1] = phi4(x[i], t[k])
            prg = swp(ap,bp,cp,dp)
            for j in range(len(y)):
                u2[i,j] = prg[j]
                u2[0,j] = phi1(y[j],t[k])
                u2[-1,j] = phi2(y[j],t[k])
        for i in range(len(x)):
            u2[i,0] = phi3(x[i],t[k])
            u2[i, -1] = phi4(x[i], t[k])
        for i in range(len(x)):
            for j in range(len(y)):
                u[i,j,k] = u2[i,j]
    return u

# График
def plot(u,analytic,x,y,t1,t2):
    fig = plt.figure(figsize=(15,10))
    x1,y2 = np.meshgrid(x, y)

    pnt = fig.add_subplot(2, 2, 1, projection='3d')
    plt.title(f'lab8 t={t[t1]}')
    pnt.plot_surface(x1,y2,u[:, :, t1],cmap=cm.RdYlGn)
    pnt.view_init(30,50)
    pnt = fig.add_subplot(2, 2, 2, projection='3d')
    plt.title(f'analytic t={t[t1]}')
    pnt.plot_surface(x1,y2,analytic[:, :, t1],cmap=cm.RdYlGn)
    pnt.view_init(30,50)

    pnt = fig.add_subplot(2, 2, 3, projection='3d')
    plt.title(f'lab8 t={t[t2]}')
    pnt.plot_surface(x1,y2,u[:, :, t2],cmap=cm.RdYlGn)
    pnt.view_init(30,50)
    pnt = fig.add_subplot(2, 2, 4, projection='3d')
    plt.title(f'analytic t={t[t2]}')
    pnt.plot_surface(x1,y2,analytic[:, :, t2],cmap=cm.RdYlGn)
    pnt.view_init(30,50)

    plt.show()

# Main
u1 = MPN_MDS(1)
u2 = MPN_MDS(2)
ext = analytic(x,y,t)
plot(u1,ext,x,y,-1,25)
plot(u2,ext,x,y,-1,25)
