import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.stats import norm
from scipy.spatial import distance
import scipy as sp
from scipy.integrate import odeint,quad,simps
from math import sqrt,sin,pi,cos,exp,log
from scipy.spatial import KDTree

class Koopman_estimation:
    def __init__(self,X):
        self.X = X #Dataset
        self.n = np.shape(self.X)[0] #Number of observations
        self.m = np.shape(self.X)[1] #Number of dimensions
        
    
