import copy

import numpy as np
from matplotlib import pyplot as plt
import warnings

Nx = 10  # количество отрезков x
Ny = 10  # количество отрезков y
lx = np.pi / 2  # правая граница координат x
ly = np.pi / 2  # правая граница координат y
hx = lx / Nx  # шаг сетки по x
hy = ly / Ny  # шаг сетки по y
bx = 2  #
by = 0  #
c = 3  #
eps = 0.001  # точность вычислений
max_iterations = 1000  # предельное количество итераций


# аналитическое решение
def analytic_solution(x, y):
    return np.exp(-x) * np.cos(x) * np.cos(y)


# условия
def phi1(y):
    return np.cos(y)


def phi2(y):
    return 0


def phi3(x):
    return np.exp(-x) * np.cos(x)


def phi4(x):
    return 0


# метод простых итераций (метод Либмана)
def liebmann_method():
    u = np.zeros((Nx + 1, Ny + 1))

    u[0, 0] = phi1(0)
    u[-1, -1] = phi2(-hy)

    for i in range(1, Nx):
        u[i, 0] = phi3(i * hx)
        u[i, -1] = phi4(i * hx)

    for j in range(1, Ny):
        u[0, j] = phi1(j * hy)
        u[-1, j] = phi2(j * hy)
        for i in range(1, Nx):
            u[i, j] = u[0, j] + (u[-1, j] - u[0, j]) / lx * i * hx

    k = 0
    while True:
        k += 1
        if k > max_iterations:
            warnings.warn("Max number of iterations exceeded!")
            break

        u_prev = copy.deepcopy(u)

        for j in range(1, Ny):
            for i in range(1, Nx):
                u[i, j] = (-(u_prev[i + 1, j] + u_prev[i - 1, j]) - hx ** 2 * (u_prev[i, j + 1] + u_prev[i, j - 1]) / (
                            hy ** 2) - bx * hx * (u_prev[i + 1, j] - u_prev[i - 1, j]) / 2 - by * hx ** 2 * (
                                       u_prev[i, j + 1] - u_prev[i, j - 1]) / (2 * hy)) / (
                                      c * hx ** 2 - 2 * (hy * hy + 1 * hx ** 2) / (hy ** 2))

        norm = np.linalg.norm(u - u_prev, np.inf)
        if norm <= eps:
            break

    print("liebmann_method: k =", k)
    return u


# метод простых итераций с верхней релаксацией, omega - параметр релаксации
def sor_method(omega):
    u = np.zeros((Nx + 1, Ny + 1))

    u[0, 0] = phi1(0)
    u[-1, -1] = phi2(-hy)

    for i in range(1, Nx):
        u[i, 0] = phi3(i * hx)
        u[i, -1] = phi4(i * hx)

    for j in range(1, Ny):
        u[0, j] = phi1(j * hy)
        u[-1, j] = phi2(j * hy)
        for i in range(1, Nx):
            u[i, j] = u[0, j] + (u[-1, j] - u[0, j]) / lx * i * hx

    k = 0
    while True:
        k = k + 1
        if k > max_iterations:
            warnings.warn("Max number of iterations exceeded!")
            break

        u_prev = copy.deepcopy(u)

        for j in range(1, Ny):
            for i in range(1, Nx):
                u[i, j] = ((-(u_prev[i + 1, j] + u[i - 1, j]) - 1 * hx ** 2 * (u_prev[i, j + 1] + u[i, j - 1]) / (
                            hy ** 2) - bx * hx * (u_prev[i + 1, j] - u[i - 1, j]) / 2 - by * hx ** 2 * (
                                        u_prev[i, j + 1] - u[i, j - 1]) / (2 * hy)) / (
                                       c * hx ** 2 - 2 * (hy ** 2 + 1 * hx ** 2) / (hy ** 2))) * omega + (1 - omega) * \
                          u_prev[i, j]

        norm = np.linalg.norm(u - u_prev, np.inf)
        if norm <= eps:
            break

    if omega == 1:
        print("seildel_method: k =", k)
    else:
        print("sor_method: k =", k)

    return u


# вывод решения в виде графиков
def show_result(y_axis, x_axis, u1, u2, u3):
    fig, ax = plt.subplots(2)
    fig.suptitle('Сравнение численных решений ДУ с аналитическим')
    fig.set_figheight(15)
    fig.set_figwidth(16)
    y = 0
    for i in range(2):
        ax[i].plot(x_axis, u1[:, y], label='Liebmann method')
        ax[i].plot(x_axis, u2[:, y], label='Seidel method')
        ax[i].plot(x_axis, u3[:, y], label='Successive over-relaxation')
        ax[i].plot(x_axis, [analytic_solution(x, y_axis[y]) for x in x_axis], label='Analytic')
        ax[i].grid(True)
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('u')
        ax[i].set_title(f'Решения при y = {y / Ny}')
        y += Ny - 1

    plt.legend(bbox_to_anchor=(1.05, 2), loc='upper right', borderaxespad=0)
    plt.show()

    fig = plt.figure(num=1, figsize=(19, 12), clear=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    fig.suptitle('Аналитическое решение')
    xgrid, ygrid = np.meshgrid(x_axis, y_axis)
    ax.plot_surface(xgrid, ygrid, analytic_solution(xgrid, ygrid))
    ax.set(xlabel='x', ylabel='y', zlabel='u')
    fig.tight_layout()
    plt.show()


# вывод изменения ошибки со временем
def show_inaccuracy(y_axis, x_axis, u):
    inaccuracy = np.zeros(Nx + 1)
    for i in range(Nx + 1):
        inaccuracy[i] = np.max(np.abs(u[i] - np.array([analytic_solution(x_axis[i], y) for y in y_axis])))

    plt.figure(figsize=(14, 8))
    plt.plot(x_axis[1:], inaccuracy[1:], 'violet', label='Ошибка')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.title('График изменения ошибки')
    plt.xlabel('y')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()


#
def main():
    u1 = liebmann_method()
    u2 = sor_method(1)  # метод Зейделя при omega = 1
    u3 = sor_method(1.5)

    y_axis = np.zeros(Ny + 1)
    for j in range(Ny + 1):
        y_axis[j] = j * hy

    x_axis = np.zeros(Nx + 1)
    for i in range(Nx + 1):
        x_axis[i] = i * hx

    show_result(y_axis, x_axis, u1, u2, u3)

    show_inaccuracy(y_axis, x_axis, u1)


if __name__ == '__main__':
    main()
