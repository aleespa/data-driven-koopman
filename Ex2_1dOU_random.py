import matplotlib.pylab as plt
import numpy as np
import scipy as sp
from scipy.integrate import trapezoid

from tools_ko.ko_estimation import KoopmanEstimation
from tools_ko.simulations import OUSimulation

x0 = 0
n = 2000000
dt = 0.01
T = dt * n
D = 1
np.random.seed(1)
Xt = OUSimulation(x0, n, dt, D)

N = 4000
X = Xt[np.random.choice(range(n), size=N), :]

K = KoopmanEstimation(X)  # Definition of the object
K.density_estimation(epsilon_0=0.2)  # Density estimation
K.bandwidth_search()  # Search for the optimal bandwidth
K.kernel_matrix()  # Computes the kernel matrix
K.KNPGenerator(
    M=400
)  # Computes the inifinitesimal generator and its eigendecomposition
K.diffusion_estimation(Xt, dt)  # Estimates the diffusion parameter

f = np.vectorize(
    lambda x: np.exp(-(x**2))
)  # Function to test the infinitesimal operator
Lf = K.infinitesimal_operator(f)

p_0 = sp.stats.norm.pdf(
    X, 1, 1
)  # Initial Distribution to apply Perron-Frobenius operator
K.transition_estimation(p_0)

f = np.vectorize(
    lambda x: np.cos(x * 2 * np.pi)
)  # Function to apply the Koopman operator
Koopman = K.koopman_operator(f)

# Plot for the eigenvalues and eigenfunctions
for j in range(K.m):
    fig, axs = plt.subplots(1, 2, dpi=100)
    axs[0].plot(K.l)
    axs[0].set_title("Eigenvalues")
    axs[0].grid(alpha=0.3)

    for i in range(min(10, 500)):
        axs[1].plot(np.sort(K.X[:, j]), K.U[np.argsort(K.X[:, j]), i], lw=1)
    axs[1].grid(alpha=0.3)
    axs[1].set_ylim(-4, 4)
    axs[1].set_xlabel(f"$X_{j+1}$")
    axs[1].set_ylabel(f"$\\varphi(x)$")
    axs[1].set_title(f"Eigenvectors")
    plt.show()

# Plot of the Infinitesimal operator
plt.figure()
plt.plot(np.sort(X[:, 0]), Lf[np.argsort(X[:, 0])], lw=2, color="r")
plt.grid(alpha=0.3)
plt.xlabel("$x$")
plt.ylabel("$\\mathcal{L}f(x)$")
plt.show()

# Plots for the evolution of the initial distribution
X_sort = np.sort(np.ravel(X))
dx = np.diff(X_sort)
fig, axs = plt.subplots(1, 1)
for t in np.arange(1, 100, 12):
    est_density = K.p_est(X.T) * np.sum(K.phi * K.csol[t], axis=1)

    est_density = est_density[np.argsort(np.ravel(X))]
    cum = trapezoid(x=X_sort, y=est_density.T)
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

# Plot for the Koopman operator
fig, ax = plt.subplots(1, 1)
for j in range(Koopman.shape[0]):
    ax.plot(
        np.sort(X[:, 0]),
        Koopman[j, np.argsort(X[:, 0])],
        color=plt.cm.inferno(j / (Koopman.shape[0])),
    )
# ax.set_xlim(-2,2)
ax.grid(alpha=0.3)
ax.set_xlabel("$x$")
ax.set_ylabel("$\\mathcal{K}^{t}\\exp(x)$")
fig.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.inferno),
    ax=ax,
    label="Time, t",
)
plt.show()
