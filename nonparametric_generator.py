import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.stats import norm
from scipy.spatial import distance
import scipy as sp
from scipy.integrate import odeint,quad,simps
from math import sqrt,sin,pi,cos,exp,log
from scipy.spatial import KDTree


def rho_0(x,X,k0=64):
    tree = KDTree(X)
    return((1/(k0-1)) * np.sum(tree.query(x,np.arange(2,k0+2))[0]**2,axis=1))

def q_0(X,Y):
    rho_l = rho_0(X,X)
    rho_i = rho_0(Y,X)
    expo = np.exp(- distance.cdist(X,Y,'sqeuclidean')/(2 * rho_l @ rho_i.T))
    return((2*np.pi ** (-d/2)/((rho_i**d) * n)) *np.sum(expo,axis=0))

def density_estimation(X,epsilon_0= None):
    """
    Estimates the density given the data 

    :param X: an nxm array 

    :returns rho: a n dimensional function 
    """
    kdensity  = sp.stats.gaussian_kde(X.T,bw_method =epsilon_0)
    return kdensity

def kernel_matrix(X,epsilon,epsilon_0 = None):
    """
    Computes the variable bandwith 
    kernel matrix given a set of observations X

    :param X: an Nxn numpy array
    """
    N,n = X.shape
    p_est = density_estimation(X,epsilon_0)
    rho = lambda x: p_est(x)**(-0.5)
    Rho = rho(X.T).reshape(1,N)
    Norm = -distance.squareform(distance.pdist(X, 'sqeuclidean'))/(4 *(Rho.T @ Rho) * epsilon)
    return(np.exp(Norm))

def Manifold_volume_kernel(Norm,epsilon,N):
    """
    Estimation of the volume of the manifold
    from a kernel and the bandwidth

    :param Norm: an nxn numpy array
    :param epsion: a float

    :return T: a float
    """
    T =  (1/N**2)*np.exp(Norm / epsilon).sum()
    return(T)

def bandwidth_search(X,h = 1e-6,K=50,verbose=False,plot=False):
    """
    Computes the optimal bandwidth for the kernel
    approximation given the data X and also computes
    the intrisic dimension of the data 

    :param X: an Nxn numpy array
    :param h: perturbation parameter for the derivative

    :return epsilon: a float
    :return d: a float 
    """
    N, n = X.shape
    # Norm = np.log(kernel_matrix(X,epsilon=1))
    Norm = -distance.squareform(distance.pdist(X, 'sqeuclidean'))/(4)
    T = lambda epsilon: Manifold_volume_kernel(Norm,epsilon,N)
    a = np.zeros(K)

    for i,e in enumerate(2**np.linspace(-20,2,K)): #evaluate for valyes from 2^-30 to 2 
        # a[i] = (log(T(h+e)) - log(T(e))) / (log(e+h) - log(e))
        a[i] = (log(T(e*2)) - log(T(e))) / (log(e*2) - log(e))

    epsilon = 2 **np.linspace(-20,2,K)[np.argmax(a)]

    d = 2 * (log(T(h+epsilon)) - log(T(epsilon))) / (log(epsilon+h) - log(epsilon))
    if verbose:
        print(f"epsilon = {epsilon:.2e}")
        print(f"d = {d:.2f}")

    if plot:
        fig,axs = plt.subplots(1,2,figsize=(12,3),dpi=100)

        axs[0].plot(2**np.linspace(-20,2,K),a,zorder=1,color='#322671')
        axs[0].scatter(epsilon,max(a),color='red',zorder=2)

        axs[0].set_xlabel('$\epsilon$')
        axs[0].set_ylabel('Power Law')
        axs[0].set_xscale('log',base=2)

        axs[1].plot(2**np.linspace(-20,2,K), [T(2**l) for l in np.linspace(-20,2)],zorder=1,color='#322671')
        axs[1].scatter(epsilon,T(epsilon),s=30,color='red',zorder=2)
        axs[1].set_xlabel('$\epsilon$')
        axs[1].set_ylabel('$T(\epsilon)$')
        axs[1].set_xscale('log',base=2)
        plt.show()

    return(epsilon,d)


def  KNPGenerator(X,M,plot=False,return_extra=False,epsilon_0 = None,epsilon=None,d=None):
    """
    Computes eigenvalues and eigenvectors 
    of the infinitesimal generator using the 
    nonparametric approximation using kernels

    :param X: an nxm array
    :param M: a integer the number of eigenvectors to compute
 
    :return l: set of M eigenvalues
    :return phi: matrix of columns of eigenvectors
    """
    N, n = X.shape
    if not epsilon:
        epsilon,d = bandwidth_search(X)

    alpha = -d/4
    p_est = density_estimation(X,epsilon_0)

    rho = lambda x: p_est(x)**(-0.5)
    Rho = rho(X.T).reshape(1,N)

    K_e = kernel_matrix(X,epsilon,epsilon_0)
    q_e = (K_e.sum(axis=1) / (Rho ** d)).reshape(N,1)

    K_e_a = K_e/((q_e**alpha)@ (q_e** alpha).T)


    I = np.identity(N)
    q_e_a = K_e_a.sum(axis=1)
    Dii = np.diag(q_e_a)
    Pii = np.diag(Rho[0,:])
    Sii = Pii @ np.sqrt(Dii)
    S_1 = np.diag(1/np.diag(Sii))
    P_2 = np.diag(np.diag(Pii)**-2)
    D_1 = np.diag(np.diag(Dii)**-1)

    L_hat = (1/epsilon) * (S_1 @ K_e_a @ S_1 - P_2)
    L_e =   (1/epsilon) * P_2 @ (D_1@ K_e_a - I )

    l, U = sp.linalg.eigh(L_hat,subset_by_index =(N-M,N-1),turbo=False)
    l = l[::-1]
    U = U[:,::-1]

    phi = S_1 @ U
    phi = phi / np.linalg.norm(phi,axis=0) * np.sqrt(N)    
    U = U / np.linalg.norm(U,axis=0) * np.sqrt(N)
    
    if plot:
        for j in range(n):
            fig,axs = plt.subplots(1,2,figsize=(12,3),dpi=100)
            axs[0].plot(l)
            axs[0].set_title('Eigenvalues')
            axs[0].grid(alpha=0.3)

            for i in range(min(10,M)):
                axs[1].scatter(X[:,j],U[:,i],s=1)
            axs[1].grid(alpha=0.3)
            axs[1].set_ylim(-4,4)
            axs[1].set_xlabel(f'$X_{j+1}$')
            axs[1].set_ylabel(f'$\\varphi(x)$')
            axs[1].set_title(f'Eigenvectors')

    if return_extra:
        return(l,phi,L_e,U)
    else:
        return(l,phi)

    

if __name__ == '__main__':
    n = 1
    N = 1000
    X = np.random.normal(0,1,size=(N,n))
    l, phi = KNPGenerator(X,M=2)
    print(l)