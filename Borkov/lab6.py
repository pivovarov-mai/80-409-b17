import numpy as np, matplotlib.pyplot as plt


# a != 0
def tochn_resh(x, t, a):
    return np.exp(-t) * np.sin(x - a * t)

def left_side(t, a):
    return -np.sin(a * t)

def right_side(t, a):
    return np.sin(a * t)

def slau(a, b, c, d, s):
    P = np.zeros(s + 1)
    Q = np.zeros(s + 1)

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    k = s - 1
    for i in range(1, s):
        P[i] = -c[i] / (b[i] + a[i] * P[i - 1])
        Q[i] = (d[i] - a[i] * Q[i - 1]) / (b[i] + a[i] * P[i - 1])
    P[k] = 0
    Q[k] = (d[k] - a[k] * Q[k - 1]) / (b[k] + a[k] * P[k - 1])

    x = np.zeros(s)
    x[k] = Q[k]

    for i in range(s - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def explicit(K, t, tau, h, x, approx_st, approx_bo, a):
    N = len(x)
    U = np.zeros((K, N))
    t += tau
    for j in range(N):
        U[0, j] = np.sin(x[j])
        if approx_st == 1:
            U[1][j] = np.sin(x[j]) - a * np.cos(x[j]) * tau
        if approx_st == 2:
            U[1][j] = np.sin(x[j]) - a * np.cos(x[j]) * tau - a ** 2 * np.sin(x[j]) * tau ** 2 / 2

    for k in range(1, K - 1):
        t += tau
        for j in range(1, N - 1):
            U[k + 1, j] = 2 * U[k, j] - U[k - 1, j] + \
                          (tau ** 2) * (a ** 2) * (U[k, j + 1] - 2 * U[k, j] + U[k, j - 1]) / h ** 2

        if approx_bo == 1:
            U[k + 1, 0] = -h * left_side(t, a) + U[k + 1, 1]
            U[k + 1, N - 1] = h * right_side(t, a) + U[k + 1, N - 2]
        elif approx_bo == 2:
            U[k + 1, 0] = (2 * h * left_side(t, a) - 4 * U[k + 1, 1] + U[k + 1, 2]) / -3
            U[k + 1, N - 1] = (2 * h * right_side(t, a) + 4 * U[k + 1, N - 2] - U[k + 1, N - 3]) / 3
        elif approx_bo == 3:
            U[k + 1, 0] = (2 * tau ** 2 * (h - h ** 2 / 2) * left_side(t, a) - 2 * tau ** 2 * U[
                k + 1, 1] - h ** 2 * (2 + 3 * tau) * U[k, 0] + h ** 2 * U[k - 1, 0]) / (
                    -2 * tau ** 2 - tau ** 2 * h ** 2 - h ** 2 - 3 * h ** 2 * tau)
            U[k + 1, N - 1] = (2 * tau ** 2 * (h + h ** 2 / 2) * right_side(t, a) + 2 * tau ** 2 * U[
                k + 1, N - 2] + h ** 2 * (2 + 3 * tau) * U[k, N - 1] - h ** 2 * U[k - 1, N - 1]) / (
                    2 * tau ** 2 + tau ** 2 * h ** 2 + h ** 2 + 3 * h ** 2 * tau)

    return U


def implicit(K, t, tau, h, x, approx_st, approx_bo, a_param):
    N = len(x)
    U = np.zeros((K, N))
    t += tau
    for j in range(N):
        U[0, j] = np.sin(x[j])
        if approx_st == 1:
            U[1][j] = np.sin(x[j]) - np.sin(x[j]) * tau
        if approx_st == 2:
            U[1][j] = np.sin(x[j]) - a_param * np.cos(x[j]) * tau - a_param ** 2 * np.sin(x[j]) * tau ** 2 / 2

    for k in range(1, K - 1):
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N)
        t += tau

        for j in range(1, N - 1):
            a[j] = 1 / h ** 2 - 1 / (2 * h)
            b[j] = -2 / h ** 2 - 1 - 1 / tau ** 2 - 3 / tau
            c[j] = 1 / h ** 2 + 1 / (2 * h)
            d[j] = - (2 * U[k, j]) / tau ** 2 - (3 * U[k, j]) / tau + U[k - 1, j] / tau ** 2
        if approx_bo == 1:
            b[0] = -1 / h
            c[0] = 1 / h
            d[0] = left_side(t, a_param)

            a[N - 1] = -1 / h
            b[N - 1] = 1 / h
            d[N - 1] = right_side(t, a_param)
        elif approx_bo == 2:
            k0 = 1 / (2 * h) / c[1]
            b[0] = (-3 / (2 * h)) + a[1] * k0
            c[0] = 2 / h + b[1] * k0
            d[0] = left_side(t, a_param) + d[1] * k0

            k1 = -(1 / (h * 2)) / a[N - 2]
            a[N - 1] = (-2 / h) + b[N - 2] * k1
            b[N - 1] = (3 / (h * 2)) + c[N - 2] * k1
            d[N - 1] = right_side(t, a_param) + d[N - 2] * k1
        elif approx_bo == 3:
            b[0] = -1 - h ** 2 / 2 - h ** 2 / (2 * tau ** 2) - (3 * h ** 2) / (2 * tau)
            c[0] = 1
            d[0] = left_side(t, a_param) * (h - h ** 2 / 2) - (
                    U[k, 0] * h ** 2) / tau ** 2 + (U[k - 1, 0] * h ** 2) / (2 * tau ** 2) - (U[k, 0] * 3 * h ** 2) / (
                               2 * tau)

            a[N - 1] = -1
            b[N - 1] = 1 + h ** 2 / 2 + h ** 2 / (2 * tau ** 2) + (3 * h ** 2) / (2 * tau)
            d[N - 1] = right_side(t, a_param) * (h + h ** 2 / 2) + (U[k, N - 1] * h ** 2) / tau ** 2 \
                       - (U[k - 1, N - 1] * h ** 2) / (2 * tau ** 2) + (U[k, N - 1] * 3 * h ** 2) / (2 * tau)

        u_new = slau(a, b, c, d, N)
        for i in range(N):
            U[k + 1, i] = u_new[i]

    return U


def main(N, K, time):
    h = (np.pi - 0) / N
    tau = time / K
    x = np.arange(0, np.pi + h / 2 - 1e-4, h)
    T = np.arange(0, time, tau)
    t = 0

    while (1):
        print("Явная конечно-разностная схема - 1; неявная конечно-разностная схема - 2; выход - 0")
        method = int(input())
        if method == 0:
            break
        else:
            print("Уровень апроксимации начальных условий - 1 или 2")
            approx_st = int(input())

            print("Уровень апроксимации краевых условий:\n"
                  "1 - двухточечная аппроксимация с первым порядком\n"
                  "2 - трехточечная аппроксимация со вторым порядком\n"
                  "3 - двухточечная аппроксимация со вторым порядком")
            approx_bo = int(input())

            print("a = ")
            a = float(input())

            if method == 1:
                if tau / h ** 2 <= 1:
                    print("Условие Куррента выполнено:", tau / h ** 2, "<= 1\n")
                    U = explicit(K, t, tau, h, x, approx_st, approx_bo, a)
                else:
                    print("Условие Куррента не выполнено:", tau / h ** 2, "> 1")
                    break
            elif method == 2:
                U = implicit(K, t, tau, h, x, approx_st, approx_bo, a)

        dt = int(input("t = "))
        U_analytic = tochn_resh(x, T[dt], a)
        error = abs(U_analytic - U[dt, :])
        plt.title("Точное и численное решения задачи")
        plt.plot(x, U_analytic, label="Точное решение", color="green")
        plt.scatter(x, U[dt, :], label="Численное решение")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.text(0.2, -0.8, "Максимальная ошибка метода: " + str(max(error)))
        plt.axis([-0.2, 3.3, -1, 1.3])
        plt.grid()
        plt.legend()
        plt.show()

        plt.title("График зависимости ошибок по времени и пространству")
        error_time = np.zeros(len(T))
        for i in range(len(T)):
            error_time[i] = max(abs(tochn_resh(x, T[i], a) - U[i, :]))
        plt.plot(T, error_time, label="По времени")
        plt.plot(x, error, label="По пространству в выбранный момент времени")
        plt.legend()
        plt.grid()
        plt.show()

    return 0


N = 50
K = 7000
time = 3
main(N, K, time)
