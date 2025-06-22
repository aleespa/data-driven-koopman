# tools_ko/ko_estimation
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.spatial import distance
from scipy.integrate import simpson
from math import log


class KoopmanEstimation:
    def __init__(self, x):
        """
        To create the estimation object you only require
        the dataset as numpy array shaped as (observations,dimension)
        """
        self.X = x  # Dataset
        self.n = np.shape(self.X)[0]  # Number of observations
        self.m = np.shape(self.X)[1]  # Number of dimensions
        print(f'Dataset with {self.n:,} observations and dimension {self.m:,}')

    def density_estimation(self, epsilon_0=None):
        """
        Estimates the density given the data 

        :param X: a nxm array
        :param epsilon_0: The bandwidth parameter

        :returns rho: a n dimensional function 
        """
        self.p_est = sp.stats.gaussian_kde((self.X).T, bw_method=epsilon_0)

    def Manifold_volume_kernel(self, alternate_Norm, alternate_epsilon):
        """
        Estimation of the volume of the manifold
        from a kernel and the bandwidth

        :param Norm: an nxn numpy array
        :param epsion: a float

        :return T: a float
        """
        T = (1 / (self.n) ** 2) * np.exp(alternate_Norm / alternate_epsilon).sum()
        return (T)

    def bandwidth_search(self, h=1e-6, K=50, plot=False):
        """
        Computes the optimal bandwidth for the kernel
        approximation given the data X and also computes
        the intrisic dimension of the data 

        :param X: an Nxn numpy array
        :param h: perturbation parameter for the derivative

        :return epsilon: a float
        :return d: a float 
        """

        alternate_Norm = -distance.squareform(distance.pdist(self.X, 'sqeuclidean')) / (4)
        T = lambda alternate_epsilon: self.Manifold_volume_kernel(alternate_Norm, alternate_epsilon)
        a = np.zeros(K)

        for i, e in enumerate(2 ** np.linspace(-20, 2, K)):
            a[i] = (log(T(e * 2)) - log(T(e))) / (log(e * 2) - log(e))

        self.epsilon = 2 ** np.linspace(-20, 2, K)[np.argmax(a)]

        self.d = 2 * (log(T(h + self.epsilon)) - log(T(self.epsilon))) / (log(self.epsilon + h) - log(self.epsilon))

        print(f"epsilon = {self.epsilon:.2e}")
        print(f"d = {self.d:.2f}")

    def kernel_matrix(self, epsilon_0=None):
        """
        Computes the variable bandwith 
        kernel matrix given a set of observations X

        :param X: an Nxn numpy array
        """
        self.rho = lambda x: self.p_est(x) ** (-0.5)
        self.Rho = self.rho((self.X).T).reshape(1, -1)
        self.Norm = -distance.squareform(distance.pdist(self.X, 'sqeuclidean')) / (
                    4 * ((self.Rho).T @ (self.Rho)) * self.epsilon)
        self.K_e = np.exp(self.Norm)

    def KNPGenerator(self, M=500, plot=False, return_extra=False, epsilon_0=None, epsilon=None, d=None):
        """
        Computes eigenvalues and eigenvectors 
        of the infinitesimal generator using the 
        nonparametric approximation using kernels

        :param X: an nxm array
        :param M: a integer the number of eigenvectors to compute
    
        :return l: set of M eigenvalues
        :return phi: matrix of columns of eigenvectors
        """

        M = min(M, self.n)

        self.alpha = -self.d / 4

        q_e = ((self.K_e).sum(axis=1) / ((self.Rho) ** (self.d))).reshape(-1, 1)

        self.K_e_a = self.K_e / ((q_e ** (self.alpha)) @ (q_e ** (self.alpha)).T)

        I = np.identity(self.n)
        q_e_a = (self.K_e_a).sum(axis=1)

        Dii = np.diag(q_e_a)
        Pii = np.diag((self.Rho)[0, :])
        Sii = Pii @ np.sqrt(Dii)
        S_1 = np.diag(1 / np.diag(Sii))
        P_2 = np.diag(np.diag(Pii) ** -2)
        D_1 = np.diag(np.diag(Dii) ** -1)

        self.L_hat = (1 / self.epsilon) * (S_1 @ (self.K_e_a) @ S_1 - P_2)
        self.L_e = (1 / self.epsilon) * P_2 @ (D_1 @ (self.K_e_a) - I)

        self.l, self.U = sp.linalg.eigh(self.L_hat, subset_by_index=(self.n - M, self.n - 1))
        self.l = self.l[::-1]
        self.U = self.U[:, ::-1]

        self.phi = S_1 @ self.U
        self.phi = self.phi / np.linalg.norm(self.phi, axis=0) * np.sqrt(self.n)
        self.U = self.U / np.linalg.norm(self.U, axis=0) * np.sqrt(self.n)

    def diffusion_estimation(self, Xt, dt):
        """
        Estimation of the diffusion parameter
        using the information of the correlation
        time from the time series

        :param Xt: a numpy array with the time series
        :param dt: the time step
        """
        n = len(Xt)
        St = np.sum(Xt - Xt.mean(), axis=1)
        C_tau = np.zeros(n)
        C0 = (1 / n) * np.sum(St ** 2)

        for j in range(1, n):
            if 1 / (n - j) * sum(St[:n - j] * St[j:]) > 0:
                C_tau[j] = 1 / (n - j) * sum(St[:n - j] * St[j:])
            else:
                break

        C_tau = C_tau[C_tau > 0]
        pv = len(C_tau)
        Tc = simpson(x=np.linspace(0, n * dt, n)[:pv], y=C_tau / C0)

        S = np.sum(self.X - (self.X).mean(), axis=1)
        s1 = np.sum((1 / self.l) * (S.T @ (self.phi)) ** 2)
        s2 = np.sum((S.T @ (self.phi)) ** 2)
        self.D = - (1 / Tc) * (s1 / s2)
        print(f'Diffusion  = {self.D:.3f}')

    def infinitesimal_operator(self, f):
        """"
        Estimation of the Infinitesimal operator
        applied to user-defined function f

        :param f: a vectorized function f

        :return Lf: an array with the approximation on the obersrvations
        """

        self.phi_1 = np.linalg.pinv(self.phi)
        Y = f(self.X)
        c_coef = (self.D * (self.phi_1 @ Y)).reshape(-1)
        Lf = np.sum((c_coef * self.l) * self.phi, axis=1)
        return (Lf)

    def transition_estimation(self, p_0, tf=1, Nt=100):
        """
        Estimation of the evolution of the density
        given an initial value for the density

        :param p_0: the intial density
        :param tf: final time for the evoultion
        :param Nt: number of steps of the evolution
        """

        rho_eq = np.vectorize(lambda x: self.p_est(x)[0])
        c_0 = (1 / self.n) * np.sum(((p_0 / rho_eq(self.X))) * self.phi, axis=0)
        tarray = np.linspace(0, tf, Nt)
        self.csol = np.exp(tarray[:, None] * self.l[None, :] * self.D) * c_0

    def koopman_operator(self, f, tf=0.2, Nt=25):
        """"
        Estimation of the Koopman operator
        applied to user-defined function f

        :param f: a vectorized function f

        :return Lf: an array with the approximation on the obersrvations
        """

        tarray = np.linspace(0, tf, Nt)
        c_0 = self.phi.T  # Initial condition
        eigen_exp = np.exp(tarray[:, None] * self.l[None, :] * self.D)
        csol = (eigen_exp[..., None] * c_0[None, ...])

        Y = f(self.X).reshape(-1)
        a = (1 / self.n) * (Y @ self.phi)

        Koopman = np.tensordot(csol, a, axes=(1, 0))

        return (Koopman)
