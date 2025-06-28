# data_driven_koopman/ko_estimation
from typing import Optional

import numpy as np
import scipy as sp
from scipy.integrate import simpson
from scipy.spatial import distance
from scipy.stats import gaussian_kde


class KoopmanEstimator:
    p_est: gaussian_kde
    epsilon: np.ndarray
    d: float
    rho: np.ndarray
    norm: np.ndarray
    k_e: np.ndarray
    diffusion_parameter: float


    def __init__(self, x: np.ndarray):
        """
        To create the estimation object, you only require
        the dataset as a numpy array shaped as (observations, dimension)
        """
        self.x = x  # Dataset
        self.n = np.shape(self.x)[0]  # Number of observations
        self.m = np.shape(self.x)[1]  # Number of dimensions
        print(f"Dataset with {self.n:,} observations and dimension {self.m:,}")

    def estimate_density(self, epsilon_0: Optional[float] = None):
        """
        Estimates the density given the data

        :param epsilon_0: The bandwidth parameter
        :returns rho: an n dimensional function
        """
        self.p_est = gaussian_kde(self.x.T, bw_method=epsilon_0)

    def get_rho(self, x: np.ndarray):
        return self.p_est(x) ** (-0.5)

    def estimate_manifold_volume(self, norm: np.ndarray, epsilon: float):
        """
        Estimation of the volume of the manifold
        from a kernel and the bandwidth

        :param norm: an nxn numpy array
        :param epsilon: a float

        :return T: a float
        """
        return (1 / self.n**2) * np.exp(norm / epsilon).sum()

    def search_bandwidth(self, h: float = 1e-6, k: int = 50):
        """
        Computes the optimal bandwidth for the kernel
        approximation given the data X and also computes
        the intrinsic dimension of the data

        :param k:
        :param h: perturbation parameter for the derivative
        """

        alternative_norm = (
            -distance.squareform(distance.pdist(self.x, "sqeuclidean")) / 4
        )
        t = lambda alternate_epsilon: self.estimate_manifold_volume(
            alternative_norm, alternate_epsilon
        )
        a = np.zeros(k)

        for i, e in enumerate(2 ** np.linspace(-20, 2, k)):
            a[i] = (np.log(t(e * 2)) - np.log(t(e))) / (np.log(e * 2) - np.log(e))

        self.epsilon = 2 ** np.linspace(-20, 2, k)[np.argmax(a)]

        self.d = (
            2
            * (np.log(t(h + self.epsilon)) - np.log(t(self.epsilon)))
            / (np.log(self.epsilon + h) - np.log(self.epsilon))
        )

        print(f"epsilon = {self.epsilon:.2e}")
        print(f"d = {self.d:.2f}")

    def calculate_kernel_matrix(self):
        """
        Computes the variable bandwidth
        kernel matrix given a set of observations X
        """
        self.rho = self.get_rho(self.x.T).reshape(1, -1)
        self.norm = -distance.squareform(distance.pdist(self.x, "sqeuclidean")) / (
                4 * (self.rho.T @ self.rho) * self.epsilon
        )
        self.k_e = np.exp(self.norm)

    def calculate_generator(
        self,
        m=500,
    ):
        """
        Computes eigenvalues and eigenvectors
        of the infinitesimal generator using the
        nonparametric approximation using kernels

        :param m: a integer the number of eigenvectors to compute

        :return l: set of M eigenvalues
        :return phi: matrix of columns of eigenvectors
        """

        m = min(m, self.n)

        self.alpha = -self.d / 4

        q_e = (self.k_e.sum(axis=1) / (self.rho ** self.d)).reshape(-1, 1)

        self.K_e_a = self.k_e / ((q_e ** self.alpha) @ (q_e ** self.alpha).T)

        I = np.identity(self.n)
        q_e_a = self.K_e_a.sum(axis=1)

        Dii = np.diag(q_e_a)
        Pii = np.diag(self.rho[0, :])
        Sii = Pii @ np.sqrt(Dii)
        S_1 = np.diag(1 / np.diag(Sii))
        P_2 = np.diag(np.diag(Pii) ** -2)
        D_1 = np.diag(np.diag(Dii) ** -1)

        self.L_hat = (1 / self.epsilon) * (S_1 @ self.K_e_a @ S_1 - P_2)
        self.L_e = (1 / self.epsilon) * P_2 @ (D_1 @ self.K_e_a - I)

        self.l, self.U = sp.linalg.eigh(
            self.L_hat, subset_by_index=(self.n - m, self.n - 1)
        )
        self.l = self.l[::-1]
        self.U = self.U[:, ::-1]

        self.phi = S_1 @ self.U
        self.phi = self.phi / np.linalg.norm(self.phi, axis=0) * np.sqrt(self.n)
        self.U = self.U / np.linalg.norm(self.U, axis=0) * np.sqrt(self.n)

    def estimate_diffusion_parameter(self, x_t, dt):
        """
        Estimation of the diffusion parameter
        using the information of the correlation
        time from the time series

        :param x_t: a numpy array with the time series
        :param dt: the time step
        """
        n = len(x_t)
        St = np.sum(x_t - x_t.mean(), axis=1)
        C_tau = np.zeros(n)
        C0 = (1 / n) * np.sum(St**2)

        for j in range(1, n):
            if 1 / (n - j) * sum(St[: n - j] * St[j:]) > 0:
                C_tau[j] = 1 / (n - j) * sum(St[: n - j] * St[j:])
            else:
                break

        C_tau = C_tau[C_tau > 0]
        pv = len(C_tau)
        Tc = simpson(x=np.linspace(0, n * dt, n)[:pv], y=C_tau / C0)

        S = np.sum(self.x - self.x.mean(), axis=1)
        s1 = np.sum((1 / self.l) * (S.T @ self.phi) ** 2)
        s2 = np.sum((S.T @ self.phi) ** 2)
        self.diffusion_parameter = -(1 / Tc) * (s1 / s2)
        print(f"Diffusion  = {self.diffusion_parameter:.3f}")

    def estimate_infinitesimal_operator(self, f):
        """
        Estimation of the Infinitesimal operator
        applied to user-defined function f

        :param f: a vectorized function f

        :return Lf: an array with the approximation on the obersrvations
        """

        self.phi_1 = np.linalg.pinv(self.phi)
        Y = f(self.x)
        c_coef = (self.diffusion_parameter * (self.phi_1 @ Y)).reshape(-1)
        return np.sum((c_coef * self.l) * self.phi, axis=1)

    def estimate_transition(self, p_0, tf=1, n_t = 100):
        """
        Estimation of the evolution of the density
        given an initial value for the density

        :param p_0: the intial density
        :param tf: final time for the evoultion
        :param n_t: number of steps of the evolution
        """

        rho_eq = np.vectorize(lambda x: self.p_est(x)[0])
        c_0 = (1 / self.n) * np.sum((p_0 / rho_eq(self.x)) * self.phi, axis=0)
        tarray = np.linspace(0, tf, n_t)
        self.csol = np.exp(tarray[:, None] * self.l[None, :] * self.diffusion_parameter) * c_0

    def koopman_operator(self, f, tf=0.2, n_t=25):
        """ "
        Estimation of the Koopman operator
        applied to user-defined function f

        :param f: a vectorized function f

        :return Lf: an array with the approximation on the observations
        """

        tarray = np.linspace(0, tf, n_t)
        c_0 = self.phi.T  # Initial condition
        eigen_exp = np.exp(tarray[:, None] * self.l[None, :] * self.diffusion_parameter)
        csol = eigen_exp[..., None] * c_0[None, ...]

        y = f(self.x).reshape(-1)
        a = (1 / self.n) * (y @ self.phi)

        return np.tensordot(csol, a, axes=(1, 0))