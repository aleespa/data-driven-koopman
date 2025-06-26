import matplotlib.pylab as plt
import numpy as np

from tools_ko.ko_estimation import KoopmanEstimator
from tools_ko.simulations import OUSimulation2d

n = int(2e6)
dt = 5e-3
D = 1
np.random.seed(1)
Xt = OUSimulation2d(D=D,n=n,dt=dt)

N = 10000
np.random.seed(1)
X = Xt[np.random.choice(range(n),size=N),:]

K = KoopmanEstimator(X) #Definition of the object
K.estimate_density(epsilon_0 = 0.25) #Density estimation
K.search_bandwidth() #Search for the optimal bandwidth
K.calculate_kernel_matrix() #Computes the kernel matrix
K.calculate_generator(m=400) #Computes the inifinitesimal generator and its eigendecomposition
K.estimate_diffusion_parameter(Xt, dt)  #Estimates the diffusion parameter



f = lambda x: np.exp(-np.linalg.norm(x,axis=1)**2) #Function to test the infinitesimal operator
Lf = K.estimate_infinitesimal_operator(f)

#Plot for the eigenvalues and eigenfunctions
for j in range(K.m):
    fig,axs = plt.subplots(1,2,dpi=100)
    axs[0].plot(K.l)
    axs[0].set_title('Eigenvalues')
    axs[0].grid(alpha=0.3)

    for i in range(min(10,500)):
        axs[1].plot(np.sort(K.x[:, j]), K.U[np.argsort(K.x[:, j]),i], lw=1)
    axs[1].grid(alpha=0.3)
    axs[1].set_ylim(-4,4)
    axs[1].set_xlabel(f'$X_{j+1}$')
    axs[1].set_ylabel(f'$\\varphi(x)$')
    axs[1].set_title(f'Eigenvectors')
    plt.show()

#Plot of the Infinitesimal operator

plt.figure()

plt.scatter(X[::2,0],X[::2,1],s=5,c=Lf[::2],cmap='jet')
plt.grid(alpha=0.3)

plt.show()