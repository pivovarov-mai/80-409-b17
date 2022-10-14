import numpy as np
from matplotlib import pyplot as plt
import warnings

T = 1  # верхняя граница времени
N = 60  # количество отрезков x
K = 1000  # количество отрезков t
l = np.pi / 2  # правая граница координат
a = 0.1  # коэффициент температуропроводности
c = -1
h = l / N  # пространственный шаг
tau = T / K  # шаг по времени
sigma = a * tau / (h ** 2)  # число Куранта, используется для анализа устойчивости решения


# аналитическое решение
def analytic_solution(x, t):
    return np.exp((c - a) * t) * np.sin(x)


# условия
def psi(x):
    return np.sin(x)


def phi0(t):
    return np.exp((c - a) * t)


def phi1(t):
    return np.exp((c - a) * t)


# явная конечно-разностная схема
def explicit_scheme(bound_condition):
    if sigma > 0.5:
        warnings.warn("Sigma > 0.5")

    u = np.zeros((K, N))

    for i in range(1, N):
        u[0][i] = psi(i * h)

    for k in range(1, K):
        for i in range(1, N - 1):
            u[k][i] = sigma * u[k - 1][i + 1] + (1 - 2 * sigma) * u[k - 1][i] + sigma * u[k - 1][i - 1] + c * tau * \
                      u[k - 1][i]

        if bound_condition == 1:
            u[k][0] = u[k][1] - h * phi0(k * tau)
            u[k][-1] = phi1(k * tau)
        elif bound_condition == 2:
            u[k][0] = (phi0(k * tau) + u[k][2] / (2 * h) - 2 * u[k][1] / h) * 2 * h / -3
            u[k][-1] = phi1(k * tau)
        elif bound_condition == 3:
            u[k][0] = (u[k][1] - h * phi0(k * tau) + (h ** 2 / (2 * tau) * u[k - 1][0])) / (1 + h ** 2 / (2 * tau))
            u[k][-1] = phi1(k * tau)
        else:
            warnings.warn("Bound condition not found")

    return u


# неявная конечно-разностная схема
def implicit_scheme(bound_condition):
    u = np.zeros((K, N))

    ai = np.zeros(N)
    bi = np.zeros(N)
    ci = np.zeros(N)
    di = np.zeros(N)

    for i in range(1, N):
        u[0][i] = psi(i * h)

    for k in range(1, K):
        for i in range(1, N - 1):
            ai[i] = sigma
            bi[i] = -2 * sigma + c * tau - 1
            ci[i] = sigma
            di[i] = -u[k - 1][i]

        if bound_condition == 1:
            bi[0] = -(1 + 2 * sigma - c * tau)
            ci[0] = 2 * sigma
            di[0] = -(u[k - 1][0] - 2 * a * tau * phi0(k * tau) / h)
            ai[-1] = 2 * sigma
            bi[-1] = -(1 + 2 * sigma - c * tau)
            #di[-1] = -(u[k - 1][-1] + sigma * phi1(k * tau))
            di[-1] = -phi1(k * tau)
        elif bound_condition == 2:
            bi[0] = -(1 + 2 * sigma - c * tau)
            ci[0] = 2 * sigma
            di[0] = -(u[k - 1][0] - 2 * a * tau * phi0(k * tau) / h)
            ai[-1] = 2 * sigma
            bi[-1] = -(1 + 2 * sigma - c * tau)
            #di[-1] = -(u[k - 1][-1] + sigma * phi1(k * tau))
            di[-1] = -phi1(k * tau)
        elif bound_condition == 3:
            bi[0] = -(1 + 2 * sigma - c * tau)
            ci[0] = 2 * sigma
            di[0] = -((1 - sigma) * u[k - 1][1] + sigma / 2 * u[k - 1][0]) - sigma * phi0(k * tau)
            ai[-1] = 2 * sigma
            bi[-1] = -(1 + 2 * sigma - c * tau)
            di[-1] = -phi1(k * tau)
        else:
            warnings.warn("Bound condition not found")

        u[k] = thomas_algorithm(ai, bi, ci, di)

    return u


# явно-неявная схема, при omega = 0.5 - схема Кранка-Николсона, при omega = 1 - полностью неявная схема, при
# omega = 0 - полностью явная схема
def explicit_implicit_scheme(omega, bound_condition):
    u = np.zeros((K, N))

    imp = implicit_scheme(bound_condition)
    exp = explicit_scheme(bound_condition)

    for k in range(0, K):
        for i in range(N):
            u[k][i] = imp[k][i] * omega + exp[k][i] * (1 - omega)

    return u


# вывод решения в виде графиков
def show_result(t_axis, x_axis, u1, u2, u3):
    fig, ax = plt.subplots(2)
    fig.suptitle('Сравнение численных решений ДУ с аналитическим')
    fig.set_figheight(15)
    fig.set_figwidth(16)
    time = 0
    for i in range(2):
        ax[i].plot(x_axis, u1[time, :], label='Explicit scheme')
        ax[i].plot(x_axis, u2[time, :], label='Implicit scheme')
        ax[i].plot(x_axis, u3[time, :], label='Crank-Nicholson scheme')
        ax[i].plot(x_axis, [analytic_solution(x, t_axis[time]) for x in x_axis], label='Analytic')
        ax[i].grid(True)
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('u')
        ax[i].set_title(f'Решения при t = {time / K}')
        time += K - 1

    plt.legend(bbox_to_anchor=(1.05, 2), loc='upper right', borderaxespad=0)
    plt.show()

    fig = plt.figure(num=1, figsize=(19, 12), clear=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    fig.suptitle('Аналитическое решение')
    xgrid, tgrid = np.meshgrid(x_axis, t_axis)
    ax.plot_surface(xgrid, tgrid, analytic_solution(xgrid, tgrid))
    ax.set(xlabel='x', ylabel='t', zlabel='u')
    fig.tight_layout()
    plt.show()


# вывод изменения ошибки со временем
def show_inaccuracy(t_axis, x_axis, u):
    inaccuracy = np.zeros(K)
    for i in range(K):
        inaccuracy[i] = np.max(np.abs(u[i] - np.array([analytic_solution(x, t_axis[i]) for x in x_axis])))

    plt.figure(figsize=(14, 8))
    plt.plot(t_axis[1:], inaccuracy[1:], 'violet', label='Ошибка')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.title('График изменения ошибки во времени')
    plt.xlabel('t')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()


# метод прогонки
def thomas_algorithm(a, b, c, d):
    size = len(a)
    P = np.zeros(size)
    Q = np.zeros(size)
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, size):
        s = (b[i] + a[i] * P[i - 1])
        P[i] = -c[i] / s
        Q[i] = (d[i] - a[i] * Q[i - 1]) / s

    result = np.zeros(size)
    result[-1] = Q[-1]

    for i in range(size - 2, -1, -1):
        result[i] = P[i] * result[i + 1] + Q[i]

    return result


# Основной аргумент схем - тип аппроксимации граничных условий, содержащих производные
# 1 - двухточечная аппроксимация с первым порядком
# 2 - трехточечная аппроксимация со вторым порядком
# 3 - двухточечная аппроксимация со вторым порядком
def main():
    u1 = explicit_scheme(1)
    u2 = implicit_scheme(1)
    u3 = explicit_implicit_scheme(0.5, 1)

    t_axis = np.zeros(K)
    for i in range(K):
        t_axis[i] = tau * i

    x_axis = np.zeros(N)
    for i in range(N):
        x_axis[i] = i * h

    show_result(t_axis, x_axis, u1, u2, u3)

    show_inaccuracy(t_axis, x_axis, u1)


if __name__ == '__main__':
    main()
