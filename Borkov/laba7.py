import matplotlib.pyplot as plt, numpy as np

def ux0(x):
    return x


def ux1(x):
    return 1 + x


def u0y(y):
    return y


def u1y(y):
    return 1 + y


def u(x, y):
    return x + y


class Solver:
    def __init__(self, method, relax=1.5, epsilon=0.01):
        self.psi1 = ux1
        self.psi0 = ux0
        self.phi0 = u0y
        self.phi1 = u1y
        self.lx0 = 0
        self.ly0 = 0
        self.lx1 = 1
        self.ly1 = 1
        self.eps = epsilon
        if method == "zeidel":
            self.method = self.ZeidelStep
        elif method == "simple":
            self.method = self.SimpleEulerStep
        elif method == "relaxation":
            self.method = lambda x, y, m: self.RelaxationStep(x, y, m, relax)

    def ZeidelStep(self, X, Y, M):
        return self.RelaxationStep(X, Y, M, w=1)

    def RelaxationStep(self, X, Y, M, w):
        norm = 0.0
        hx2 = self.hx * self.hx
        hy2 = self.hy * self.hy
        for i in range(1, self.Ny - 1):
            for j in range(1, self.Nx - 1):
                diff = hy2 * (M[i][j - 1] + M[i][j + 1])
                diff += hx2 * (M[i - 1][j] + M[i + 1][j])
                diff /= 2 * (hy2 + hx2 - hx2 * hy2)
                diff -= M[i][j]
                diff *= w
                M[i][j] += diff
                diff = abs(diff)
                norm = diff if diff > norm else norm
        return norm

    def SimpleEulerStep(self, X, Y, M):
        tmp = [[0.0 for _ in range(self.Nx)] for _ in range(self.Ny)]
        norm = 0.0
        hx2 = self.hx * self.hx
        hy2 = self.hy * self.hy

        for i in range(1, self.Ny - 1):
            tmp[i][0] = M[i][0]
            for j in range(1, self.Nx - 1):
                tmp[i][j] = hy2 * (M[i][j - 1] + M[i][j + 1])
                tmp[i][j] += hx2 * (M[i - 1][j] + M[i + 1][j])
                tmp[i][j] /= 2 * (hy2 + hx2 - hx2 * hy2)
                diff = abs(tmp[i][j] - M[i][j])
                norm = diff if diff > norm else norm
                tmp[i][-1] = M[i][-1]
        for i in range(1, self.Ny - 1):
            M[i] = tmp[i]
        return norm

    def set_l0_l1(self, lx0, lx1, ly0, ly1):
        self.lx0 = lx0
        self.lx1 = lx1
        self.ly0 = ly0
        self.ly1 = ly1

    def _compute_h(self):
        self.hx = (self.lx1 - self.lx0) / (self.Nx - 1)
        self.hy = (self.ly1 - self.ly0) / (self.Ny - 1)

    @staticmethod
    def nparange(start, end, step=1):
        now = start
        e = 0.00000000001
        while now - e <= end:
            yield now
            now += step

    def init_values(self, X, Y):
        ans = [[0.0 for _ in range(self.Nx)] for _ in range(self.Ny)]
        for j in range(1, self.Nx - 1):
            coeff = (self.psi1(X[-1][j]) - self.psi0(X[0][j])) / (self.ly1 - self.ly0)
            addition = self.psi0(X[0][j])
            for i in range(self.Ny):
                ans[i][j] = coeff * (Y[i][j] - self.ly0) + addition
            for i in range(self.Ny):
                ans[i][0] = self.phi0(Y[i][0])
                ans[i][-1] = self.phi1(Y[i][-1])
        return ans

    def __call__(self, Nx=10, Ny=10):
        self.Nx, self.Ny = Nx, Ny
        self._compute_h()

        x = list(self.nparange(self.lx0, self.lx1, self.hx))
        y = list(self.nparange(self.ly0, self.ly1, self.hy))
        X = [x for _ in range(self.Ny)]
        Y = [[y[i] for _ in x] for i in range(self.Ny)]
        ans = self.init_values(X, Y)

        self.itters = 0

        while (self.method(X, Y, ans) >= self.eps):
            self.itters += 1
            ans = np.array(ans)
        return X, Y, ans


def real_z(lx0, lx1, ly0, ly1, f):
    x = np.arange(lx0, lx1 + 0.001, 0.001)
    y = np.arange(ly0, ly1 + 0.001, 0.001)
    X = np.ones((y.shape[0], x.shape[0]))
    Y = np.ones((x.shape[0], y.shape[0]))
    Z = np.ones((y.shape[0], x.shape[0]))
    for i in range(Y.shape[0]):
        Y[i] = y
        Y = Y.T
    for i in range(X.shape[0]):
        X[i] = x
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = f(X[i, j], Y[i, j])
    return X, Y, Z


def plot(Nx=5, Ny=5, eps=1, method="zeidel"):
    solver = Solver(epsilon=eps, method=method, relax=1.6)
    x, y, z = solver(Nx, Ny)
    fig = plt.figure(num=1, figsize=(10, 7), clear=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_wireframe(*real_z(0, 1, 0, 1, u), color="green")
    ax.plot_surface(np.array(x), np.array(y), np.array(z), color="red")
    ax.set(xlabel='x', ylabel='y', zlabel='u', title=f"График приближения функции конечно-разностным методом (N={Nx})")
    fig.tight_layout()
    plt.show()


def error(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            ans += (f(x[i][j], y[i][j]) - z[i][j]) ** 2
    return (ans ** 0.5) / (len(z) * len(z[0]))


def get_step_error(solver, real_f):
    h = []
    e = []
    N = 10
    x, y, z = solver(N, N)
    h.append(solver.hx)
    e.append(error(x, y, z, real_f))
    N = 20
    x, y, z = solver(N, N)
    h.append(solver.hx)
    e.append(error(x, y, z, real_f))
    N = 40
    x, y, z = solver(N, N)
    h.append(solver.hx)
    e.append(error(x, y, z, real_f))
    return h, e


def error_step(eps, r):
    simp = Solver(epsilon=eps, method="simple")
    zeid = Solver(epsilon=eps, method="zeidel")
    relax = Solver(epsilon=eps, method="relaxation", relax=r)
    plt.figure(figsize=(10, 7))
    plt.title(f"Зависимость погрешности от длины шага eps={eps}, w={r}, (N=10,20,40)")
    h_s, e_s = get_step_error(simp, u)
    h_z, e_z = get_step_error(zeid, u)
    h_r, e_r = get_step_error(relax, u)
    plt.plot(h_s, e_s, label="МПИ", color="olive")
    plt.plot(h_z, e_z, label="Метод Зейделя", color="magenta")
    plt.plot(h_r, e_r, label="МПИ с верхней релаксацией", color="orange")
    plt.xlabel("h")
    plt.ylabel("error")
    plt.legend()
    plt.grid()
    plt.show()


Nx = 10
Ny = 10
r = 1.6
eps = 0.01

plot(Nx, Ny, eps, method="simple")
plot(Nx, Ny, eps, method="zeidel")
plot(Nx, Ny, eps, method="relaxation")
error_step(eps, r=r)

solver = Solver(epsilon=eps, method="simple")
solver(Nx, Ny)
print("Кол-во итераций метода простых иттераций:", solver.itters)

solver = Solver(epsilon=eps, method="zeidel")
solver(Nx, Ny)
print("Кол-во итераций метода Зейделя:", solver.itters)

solver = Solver(epsilon=eps, method="relaxation", relax=r)
solver(Nx, Ny)
print("Кол-во итераций МПИ с верхней релаксацией:", solver.itters)
