import numpy as np
import matplotlib.pyplot as plt
import math


class ParabolicTypeEquations:
    def __init__(self, func, phi_0, phi_l, xi, x, t, a, N=100):
        self.func = func
        self.phi_0 = phi_0
        self.phi_l = phi_l
        self.xi = xi
        self.x = x
        self.t = t
        self.N = N
        self.h = (x[-1] - x[1]) / N
        self.tau = (t[-1] - t[1]) / N
        self.sigma = a * a * self.tau / self.h / self.h

    # Точное решение
    def exact_solution(self):
        return np.array([[self.func(x, t) for t in self.t] for x in self.x])

    # Явная конечно-разностная схема
    def explicit_finite_difference_scheme(self):
        u = [[0 for j in range(len(self.t))] for i in range(len(self.x))]

        for j in range(len(self.x)):
            u[j][0] = self.xi(self.x[j])

        for k in range(len(self.t) - 1):
            u[0][k + 1] = self.phi_0(self.t[k + 1])

        for k in range(len(self.t) - 1):
            u[-1][k + 1] = self.phi_l(self.t[k + 1])

        for j in range(1, len(self.x) - 1):
            for k in range(len(self.t) - 1):
                u[j][k + 1] = self.sigma * u[j + 1][k] + (1 - 2 * self.sigma) * u[j][k] + self.sigma * u[j - 1][k]

        return u

    # Неявная схема
    def implicit_finite_difference_scheme(self):
        u = [[0 for j in range(len(self.t))] for i in range(len(self.x))]

        for k in range(len(self.t) - 1):
            u[0][k + 1] = self.phi_0(self.t[k + 1])

        for k in range(len(self.t) - 1):
            u[-1][k + 1] = self.phi_l(self.t[k + 1])

        for j in range(len(self.x)):
            u[j][0] = self.xi(self.x[j])

        for k in range(len(self.t) - 1):
            A = [self.sigma if i != 0 else 0 for i in range(self.N - 2)]
            B = [-(1 + 2 * self.sigma) for i in range(self.N - 2)]
            C = [self.sigma if i != self.N - 3 else 0 for i in range(self.N - 2)]
            D = [-u[j][k] for j in range(2, self.N - 2)]
            D.insert(0, -(u[1][k] + self.sigma * self.phi_0(self.t[k + 1])))
            D.append(-(u[self.N - 1][k] + self.sigma * self.phi_l(self.t[k + 1])))

            uk = self.progonka(A, B, C, D)
            for j in range(1, self.N - 1):
                u[j][k + 1] = uk[j - 1]

        return u

    # Явно-неявня схема
    def explicit_implicit_finite_difference_scheme(self, teta):
        u = [[0 for j in range(len(self.t))] for i in range(len(self.x))]

        for k in range(len(self.t) - 1):
            u[0][k + 1] = self.phi_0(self.t[k + 1])

        for k in range(len(self.t) - 1):
            u[-1][k + 1] = self.phi_l(self.t[k + 1])

        for j in range(len(self.x)):
            u[j][0] = self.xi(self.x[j])

        for k in range(len(self.t) - 1):
            A = [teta * self.sigma if i != 0 else 0 for i in range(self.N - 2)]
            B = [-(1 + 2 * teta * self.sigma) for i in range(self.N - 2)]
            C = [teta * self.sigma if i != self.N - 3 else 0 for i in range(self.N - 2)]
            D = [- (1 - teta) * self.sigma * u[j + 1][k] - (1 - 2 * (1 - teta) * self.sigma) * u[j][k] - (
                        1 - teta) * self.sigma * u[j - 1][k] for j in range(2, self.N - 2)]
            D.insert(0, -(teta * self.sigma * self.phi_0(self.t[k + 1]) + (1 - teta) * self.sigma * u[2][k] + (
                        1 - 2 * (1 - teta) * self.sigma) * u[1][k] + (1 - teta) * self.sigma * self.phi_0(self.t[k])))
            D.append(-(teta * self.sigma * self.phi_l(self.t[k + 1]) + (1 - teta) * self.sigma * self.phi_l(self.t[k]) + (
                        1 - 2 * (1 - teta) * self.sigma) * u[self.N - 1][k] + (1 - teta) * self.sigma * u[self.N - 2][
                           k]))

            uk = self.progonka(A, B, C, D)

            for j in range(1, self.N - 1):
                u[j][k + 1] = uk[j - 1]

        return u

    def progonka(self, a, b, c, d):
        n = len(a)
        for i in range(n):
            if math.fabs(b[i]) < math.fabs(a[i]) + math.fabs(c[i]):
                raise Exception

        #   Формирование массивов P, Q (Расчет значений) ((Прямой ход))

        P, Q = [-c[0] / b[0]], [d[0] / b[0]]

        for i in range(1, n):
            P.append(-c[i] / (b[i] + a[i] * P[i - 1]))
            Q.append((d[i] - a[i] * Q[i - 1]) / (b[i] + a[i] * P[i - 1]))

        #   Вычисление решения системы (Обратный ход)
        x = [Q[n - 1]]
        for i in range(1, n):
            x.append(P[n - 1 - i] * x[i - 1] + Q[n - 1 - i])

        return list(reversed(x))


def main():
    a = 0.01

    U = lambda x, t: np.exp(-a * t) * np.cos(x)

    phi_0 = lambda t: math.exp(-a * t)
    phi_l = lambda t: -math.exp(-a * t)
    xi = lambda x: math.cos(x)

    N = 100
    X = np.linspace(0, 3, N)
    T = np.linspace(0, math.pi, N)

    equations = ParabolicTypeEquations(U, phi_0, phi_l, xi, X, T, a)

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    ax.set_title('Точное решение')

    u = equations.exact_solution()

    Q, W = np.meshgrid(X, T)
    ax.plot_surface(W, Q, np.array(u))

    ax.set_xlabel('x Label')
    ax.set_ylabel('t Label')
    ax.set_zlabel('u Label')

    ax = fig.add_subplot(1, 4, 2, projection='3d')
    ax.set_title('Явная схема')

    u = equations.explicit_finite_difference_scheme()

    Q, W = np.meshgrid(X, T)
    ax.plot_surface(W, Q, np.array(u))

    ax.set_xlabel('x Label')
    ax.set_ylabel('t Label')
    ax.set_zlabel('u Label')
    print('Явная схема: средкв ошибка:',
          math.sqrt(sum([sum([(U(X[i], T[j]) - u[i][j]) ** 2 for j in range(len(T))]) for i in range(len(X))])))

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    ax.set_title('Неявная схема')

    u = equations.implicit_finite_difference_scheme()

    Q, W = np.meshgrid(X, T)
    ax.plot_surface(W, Q, np.array(u))

    ax.set_xlabel('x Label')
    ax.set_ylabel('t Label')
    ax.set_zlabel('u Label')
    print('Неявная схема: средкв ошибка:',
          math.sqrt(sum([sum([(U(X[i], T[j]) - u[i][j]) ** 2 for j in range(len(T))]) for i in range(len(X))])))

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    ax.set_title('Явно-неявная схема')

    u = equations.explicit_implicit_finite_difference_scheme(1 / 2)

    Q, W = np.meshgrid(X, T)
    ax.plot_surface(W, Q, np.array(u))

    ax.set_xlabel('x Label')
    ax.set_ylabel('t Label')
    ax.set_zlabel('u Label')
    print('Явно-неявная схема: средкв ошибка:',
          math.sqrt(sum([sum([(U(X[i], T[j]) - u[i][j]) ** 2 for j in range(len(T))]) for i in range(len(X))])))


if __name__ == "__main__":
    main()
