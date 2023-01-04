import copy
import numpy as np
from matplotlib import pyplot as plt, cm

lx = 1
ly = np.pi/2
nx = 10
ny = 10
hx = lx/nx
hy = ly/ny
eps = 10**(-3)

#Дано

def phi1(y):
    return np.cos(y)

def phi2(y):
    return np.cos(y)*np.exp(1)

def phi3(x):
    return 0

def phi4(x):
    return -np.exp(x)

def exact(x,y):
    return np.exp(x) * np.cos(y)

x = np.arange(0, lx+hx, hx)
y = np.arange(0, ly+hy, hy)

def u_zero():
    u = np.zeros(shape=(len(x), len(y)))
    for j in range(0,len(y)):
        u[0][j] = phi1(y[j])
        u[-1][j] = phi2(y[j])
    for j in range(len(y) - 1, -1, -1):
        for i in range(len(x) - 2, 0, -1):
            u[i, j] = u[i + 1, j] * y[i] / (x[j] + y[i])
    for j in range(len(y) - 1, -1, -1):
        for i in range(1,len(x)-1):
            u[i, j] += u[i + 1, j] * y[i] / (x[j] + y[i])
            u[i, j] /= 2
    return u

# Метод Либмана

def Lib_method():
    u = u_zero()
    v = np.zeros(shape=(len(x), len(y)))
    k = 0
    m = np.abs(u-v).max()
    while m > eps:
        v = copy.deepcopy(u)
        for i in range(1, len(x)-1):
            u[i][0] = v[i][1]-phi3(x[i])*hy
            u[i][-1] = v[i][-2]+phi4(x[i])*hy
        for i in range(1, len(x)-1):
            for j in range(1, len(y)-1):
                u[i,j] = (v[i+1][j] + v[i-1][j] + v[i][j-1] + v[i][j+1]) / 4
        m = np.abs(u-v).max()
        k += 1
    print("Метод Либмана")
    print("Шагов: ",k)
    print("Погрешность: ",m)
    return u

# Метод верхней релаксации

def relax(w):
    u = u_zero()
    v = np.zeros(shape=(len(x), len(y)))
    k = 0
    m = np.abs(u-v).max()
    while m > eps:
        v = copy.deepcopy(u)
        for i in range(1, len(x)-1):
            u[i][0] = v[i][1]-phi3(x[i])*hy
            u[i][-1] = v[i][-2]+phi4(x[i])*hy
        for i in range(1, len(x)-1):
            for j in range(1, len(y)-1):
                u[i,j] += w * ((u[i-1][j] + v[i+1][j] + u[i][j-1] + v[i][j+1]) / 4 - v[i,j])
        m = np.abs(u-v).max()
        k += 1
    if (w == 1):
        print("Метод Зейделя")
    else:
        print("Метод верхней релаксации")
    print("Шагов: ",k)
    print("Погрешность: ",m)
    return u

def analytic(x, y):
    ext = np.zeros((len(x), len(y)))
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            ext[i, j] = exact(x[i], y[j])
    return ext

def plot(mtd, analytic, x, y):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    x1, y1 = np.meshgrid(x, y)
    pnt = fig.add_subplot(1, 2, 1, projection='3d')
    plt.title('lab7')
    pnt.set_xlabel('x', fontsize=15)
    pnt.set_ylabel('y', fontsize=15)
    pnt.set_zlabel('u', fontsize=15)
    pnt.plot_surface(x1, y1, mtd, cmap=cm.RdYlGn)
    pnt = fig.add_subplot(1, 2, 2, projection='3d')
    plt.title('analytic')
    pnt.set_xlabel('x', fontsize=15)
    pnt.set_ylabel('y', fontsize=15)
    pnt.set_zlabel('u', fontsize=15)
    pnt.plot_surface(x1, y1, analytic, cmap=cm.RdYlGn)

    plt.show()

# Main
analytic = analytic(x, y)

lm = Lib_method() # Метод Либмана
plot(lm,analytic, x, y)

zeyd = relax(1) # Метод Зейделя
plot(zeyd,analytic, x, y)

rl = relax(1.5) # Метод верхней релаксации
plot(rl,analytic, x, y)
