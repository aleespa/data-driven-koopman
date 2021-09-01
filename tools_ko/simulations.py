#tools_ko/simulations
import numpy as np
from scipy.special import erfinv

def OUSimulation(x0, n,dt,D):
    """
    Function to create the trayectory of a OU process
    with difussion paramter D 

    :param x0: Intial position
    :param n:  Number of steps
    "param dt: Step size
    :param D:  Difussion parameter

    :return X: Trajectory
    """
    X = np.zeros((n,1))
    noise = np.random.normal(loc=0,scale=np.sqrt(dt),size=(n,1))
    for i in range(1,n):
        x = X[i-1,:]
        X[i,:] = x - x*dt +np.sqrt(2*D) * noise[i,:]
    return(X)

def OUDeterministic(N):
    """
    Determinisic sampling from the 
    inviariand distribution of the OU 
    process

    :param N: Number of sampling points
    """
    x_hat = np.linspace(0,1,N+2)
    X = np.sqrt(2)*erfinv(2*x_hat[1:-1] - 1)
    return(X)

def DWSimulation(x0, n,dt,D):
    """
    Function to create the trayectory of a OU process
    with difussion paramter D 

    :param x0: Intial position
    :param n:  Number of steps
    "param dt: Step size
    :param D:  Difussion parameter

    :return X: Trajectory
    """
    DV = lambda x:  5*x**5 - 10*x**3
    t = np.linspace(x0,dt*n,n)
    X = np.zeros((n,1))
    noise = np.random.normal(loc=0,scale=np.sqrt(dt),size=(n,1))
    for i in range(1,n):
        x = X[i-1,:]
        X[i,:] = x -DV(x)*dt +np.sqrt(2*D) * noise[i,:]
    return(X)

def OUSimulation2d(n=100000,dt = 0.0001,D=1):
    """
    Function to create the trayectory of a 2d OU process
    with difussion paramter D starting at (0,0)

    :param n:  Number of steps
    :param dt: Step size
    :param D:  Difussion parameter

    :return Z: Trajectory
    """
    W = np.random.normal(loc=0,scale=np.sqrt(dt),size=(n,2))
    Z = np.zeros((n,2))
    Z[0,...] = 0
    for i in range(1,n):
        Z_1 = Z[i-1,...]
        Z[i,...] = Z_1 - Z_1*dt + np.sqrt(2*D) * W[i,:]
    return(Z)



def QWSimulation2d(n=100000,dt = 0.0001,D=1):
    """
    Function to create the trayectory of a quadruple well
    processwith difussion paramter D starting at (0,0)

    :param n:  Number of steps
    :param dt: Step size
    :param D:  Difussion parameter

    :return Z: Trajectory
    """
    def dV(X,k=4):
        r = np.linalg.norm(X,axis=-1)
        phi = np.arctan2(X[...,1],X[...,0]) 
        dx1 = np.cos(phi) * (20*(r-1) - 1/r**2) - (1/r)*np.sin(phi)*(-k*np.sin(k*phi) +0.5*((1/np.cos(0.5 *phi))*np.tan(0.5 *phi)))
        dx2 = np.sin(phi) * (20*(r-1) - 1/r**2) + (1/r)*np.cos(phi)*(-k*np.sin(k*phi) +0.5*((1/np.cos(0.5 *phi))*np.tan(0.5 *phi)))
        return(np.array([dx1,dx2]))

    W = np.random.normal(loc=0,scale=np.sqrt(dt),size=(n,2))
    Z = np.zeros((n,2))
    Z[0,...] = 1
    for i in range(1,n):
        Z_1 = Z[i-1,...]
        Z[i,...] = Z_1 - dV(Z_1)*dt + np.sqrt(2*D) * W[i,:]
    return(Z)