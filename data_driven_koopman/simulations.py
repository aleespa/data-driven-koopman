# data_driven_koopman/simulations
import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.special import erfinv


def simulate_ou_process(x0, n, dt, diffusion):
    """
    Function to create the trajectory of an OU process
    with diffusion parameter D

    :param x0: Initial position
    :param n:  Number of steps to simulate
    :param dt: Step size
    :param diffusion:  Diffusion parameter

    :return X: Trajectory
    """
    z = np.zeros((n, 1))
    noise = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n, 1))
    for i in range(1, n):
        x = z[i - 1, :]
        z[i, :] = x - x * dt + np.sqrt(2 * diffusion) * noise[i, :]
    return z


def generate_deterministic_ou_process(n):
    """
    Deterministic sampling from the
    invariant distribution of the OU
    process

    :param n: Number of sampling points
    """
    x_hat = np.linspace(0, 1, n + 2)
    X = np.sqrt(2) * erfinv(2 * x_hat[1:-1] - 1)
    return X


def simulate_double_well(x0, n, dt, D):
    """
    Function to create the trayectory of a OU process
    with difussion paramter D

    :param x0: Intial position
    :param n:  Number of steps
    "param dt: Step size
    :param D:  Difussion parameter

    :return X: Trajectory
    """
    DV = lambda x: 5 * x**5 - 10 * x**3
    t = np.linspace(x0, dt * n, n)
    X = np.zeros((n, 1))
    noise = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n, 1))
    for i in range(1, n):
        x = X[i - 1, :]
        X[i, :] = x - DV(x) * dt + np.sqrt(2 * D) * noise[i, :]
    return X


def generate_deterministic_double_well(N, D=1):
    """
    Determinisic sampling from the
    inviariant distribution of the
    Double-Well potential process

    :param N: Number of sampling points
    """
    V = lambda x: (5 / 6) * x**6 - (5 / 2) * x**4
    Z = trapezoid(
        x=np.linspace(-10, 10, 2000),
        y=[np.exp(-V(x) / D) for x in np.linspace(-10, 10, 2000)],
    )
    rho_inf = lambda x: (1 / Z) * np.exp(-V(x) / D)
    x = np.linspace(-4, 4, 10000)
    F = cumulative_trapezoid(x=x, y=rho_inf(x))
    x_bar = np.linspace(1e-10, 1 - 1e-10, N)
    X = np.zeros((N, 1))
    for i in range(N):
        X[i] = (x[:-1][F > x_bar[i]])[0]
    return X


def simulate_2d_ou_process(n=100000, dt=0.0001, D=1):
    """
    Function to create the trayectory of a 2d OU process
    with difussion paramter D starting at (0,0)

    :param n:  Number of steps
    :param dt: Step size
    :param D:  Difussion parameter

    :return Z: Trajectory
    """
    W = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n, 2))
    Z = np.zeros((n, 2))
    Z[0, ...] = 0
    for i in range(1, n):
        Z_1 = Z[i - 1, ...]
        Z[i, ...] = Z_1 - Z_1 * dt + np.sqrt(2 * D) * W[i, :]
    return Z


def simulate_2d_quadruple_well(n=100000, dt=0.0001, D=1):
    """
    Function to create the trayectory of a quadruple well
    processwith difussion paramter D starting at (0,0)

    :param n:  Number of steps
    :param dt: Step size
    :param D:  Difussion parameter

    :return Z: Trajectory
    """

    def dV(X, k=4):
        r = np.linalg.norm(X, axis=-1)
        phi = np.arctan2(X[..., 1], X[..., 0])
        dx1 = np.cos(phi) * (20 * (r - 1) - 1 / r**2) - (1 / r) * np.sin(phi) * (
            -k * np.sin(k * phi) + 0.5 * ((1 / np.cos(0.5 * phi)) * np.tan(0.5 * phi))
        )
        dx2 = np.sin(phi) * (20 * (r - 1) - 1 / r**2) + (1 / r) * np.cos(phi) * (
            -k * np.sin(k * phi) + 0.5 * ((1 / np.cos(0.5 * phi)) * np.tan(0.5 * phi))
        )
        return np.array([dx1, dx2])

    W = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n, 2))
    Z = np.zeros((n, 2))
    Z[0, ...] = 1
    for i in range(1, n):
        Z_1 = Z[i - 1, ...]
        Z[i, ...] = Z_1 - dV(Z_1) * dt + np.sqrt(2 * D) * W[i, :]
    return Z
