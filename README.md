# Data-driven approximation of the Koopman operator for a SDE
Repository for my MSc project in the data-driven approximation of the Koopman operator for a SDE, the method is based in the paper "Nonparametric uncertainty quantification for
stochastic gradient flow" by Tyrus Berry and John Harlim [Link](https://epubs.siam.org/doi/10.1137/14097940X)

The main class is `ko_estimation.py`
- `density_estimation` Estimation of the density for the dataset provided 
- `bandwidth_search` Search of the optimal bandwidth for the Gaussian Kernel
- `kernel_matrix` Computation of the Kernel matrix given the optimal bandwidth computed
- `KNPGenerator` Computes the approximation of the infinitesimal operator and its eigenvectors, 
- `diffusion_estimation` Approximates the diffusion term using the trajectory of the time-series
- `infinitesimal_operator` Applies the infinitesimal operator to a given function
- `transition_estimation` Evolves an initial density function solving the Fokker-Plank equation
- `koopman_operator` Computes the time dependent Koopman Operator for an observable.
# References
- Tyrus Berry and John Harlim. Nonparametric uncertainty quantification for stochastic gradient flows. SIAM/ASA Journal on Uncertainty Quantification, 3(1):484–508, 2015.[Link](https://doi.org/10.1137/14097940X)
- Tyrus Berry and John Harlim. Variable bandwidth diffusion kernels. Applied and Computational Harmonic Analysis, 40(1):68–96, jan 2016. [Link](https://doi.org/10.1016/j.acha.2015.01.001)