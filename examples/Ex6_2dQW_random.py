import matplotlib.pylab as plt
import numpy as np

from tools_ko.ko_estimation import KoopmanEstimation
from tools_ko.simulations import QWSimulation2d

#Simulation of the trayectory
n = int(2e7)
dt = 1e-3
D = 1
np.random.seed(1)
Xt = QWSimulation2d(D=D,n=n,dt=dt)
Xt = Xt[np.linalg.norm(Xt,axis=1)<10,:]

#Subsample from the trajectory
N = 10000
np.random.seed(1)
X = Xt[np.random.choice(range(n), size=N),:]

K = KoopmanEstimation(X) #Definition of the object
K.density_estimation(epsilon_0 = 0.15) #Density estimation
K.bandwidth_search() #Search for the optimal bandwidth
K.kernel_matrix() #Computes the kernel matrix
K.KNPGenerator(M=400) #Computes the inifinitesimal generator and its eigendecomposition 
K.diffusion_estimation(Xt,dt)  #Estimates the diffusion parameter



f = lambda x:  x[:,1] + x[:,0] #Function to test the infinitesimal operator
Lf = K.infinitesimal_operator(f)

#Plot for the eigenvalues and eigenfunctions
for j in range(K.m):
    fig,axs = plt.subplots(1, 2, dpi=100)
    axs[0].plot(K.l)
    axs[0].set_title('Eigenvalues')
    axs[0].grid(alpha=0.3)

    for i in range(min(10,500)):
        axs[1].plot(np.sort(K.X[:,j]),K.U[np.argsort(K.X[:,j]),i],lw=1)
    axs[1].grid(alpha=0.3)
    axs[1].set_ylim(-4, 4)
    axs[1].set_xlabel(f'$X_{j+1}$')
    axs[1].set_ylabel(f'$\\varphi(x)$')
    axs[1].set_title(f'Eigenvectors')
    plt.show()

#Plot of the Infinitesimal operator

plt.figure()

plt.scatter(X[::2,0],X[::2,1],s=5,c=Lf[::2],cmap='jet')
plt.grid(alpha=0.3)

plt.show()