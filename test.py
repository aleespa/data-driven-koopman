import matplotlib.pylab as plt
import numpy as np
import scipy as sp
from scipy.integrate import cumulative_trapezoid

from data_driven_koopman.ko_estimation import KoopmanEstimator
from data_driven_koopman.simulations import simulate_ou_process

x0 = 0
n = 20000
dt = 0.01
T = dt * n
D = 1
np.random.seed(1)
Xt = simulate_ou_process(x0, n, dt, D)

N = 2000
X = Xt[np.random.choice(range(n), size=N), :]

K = KoopmanEstimator(X)
K.estimate_density(epsilon_0=0.2)
K.search_bandwidth()
K.calculate_kernel_matrix()
K.calculate_generator(m=30)
K.estimate_diffusion_parameter(Xt, dt)

f = np.vectorize(lambda x: np.exp(-(x**2)))

Lf = K.estimate_infinitesimal_operator(f)

p_0 = sp.stats.norm.pdf(X, 1, 1)

K.estimate_transition(p_0)


for j in range(K.m):
    fig, axs = plt.subplots(1, 2, figsize=(12, 3), dpi=100)
    axs[0].plot(K.l)
    axs[0].set_title("Eigenvalues")
    axs[0].grid(alpha=0.3)

    for i in range(min(10, 500)):
        axs[1].plot(np.sort(K.x[:, j]), K.U[np.argsort(K.x[:, j]), i], lw=1)
    axs[1].grid(alpha=0.3)
    axs[1].set_ylim(-4, 4)
    axs[1].set_xlabel(f"$X_{j+1}$")
    axs[1].set_ylabel(f"$\\varphi(x)$")
    axs[1].set_title(f"Eigenvectors")
    plt.show()

X_sort = np.sort(np.ravel(X))
dx = np.diff(X_sort)
fig, axs = plt.subplots(1, 1, figsize=(3, 2))

for t in np.arange(1, 100, 12):
    est_density = K.p_est(X.T) * np.sum(K.phi * K.csol[t], axis=1)

    est_density = est_density[np.argsort(np.ravel(X))]
    cum = cumulative_trapezoid(x=X_sort, y=est_density.T, initial=0)
    plt.plot(X_sort, (1 / cum) * est_density, lw=1, color=plt.cm.bwr(t / 100), ls="--")


plt.plot(
    np.linspace(min(X), max(X), 100),
    K.p_est(np.linspace(min(X), max(X), 100).T),
    color="r",
    lw=1,
)
plt.plot(
    np.linspace(min(X), max(X), 100),
    sp.stats.norm.pdf(np.linspace(min(X), max(X), 100), 1, 1),
    color="k",
    lw=1,
)
plt.xlabel("$x$")
plt.ylabel("$p(t,x)$")
plt.grid(alpha=0.3)
plt.show()

plt.figure()
plt.scatter(X, Lf, s=2, color="k")
plt.show()
