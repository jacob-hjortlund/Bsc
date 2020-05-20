import numpy as np
import log_bessel as lb
from scipy.linalg import cholesky, inv
from scipy.special import loggamma as lg
from scipy.special import gamma

def rbf(theta,x):
    
    return theta[0]**2 * np.exp(-0.5 * theta[1]**-2 * np.subtract.outer(x,x)**2)

def periodic(theta,x):
    
    return theta[0]**2 * np.exp(-2 * theta[1]**-2 * np.sin(np.pi * np.subtract.outer(x,x) / theta[2])**2)

def local_periodic(theta,x):
    
    return rbf(theta,x) * periodic(theta,x)

def matern(theta, x):

    u = np.sqrt(2 * theta[1]) * np.abs(np.subtract.outer(x,x) / theta[2])
    u[u == 0.0] += np.finfo(float).eps

    log_K = 2*np.log(theta[0]) + (1-theta[1])*np.log(2) - lg(theta[1]) + theta[1]*np.log(u) + lb.log_bessel_k(theta[1], u)

    return np.exp(log_K)

def power_law(theta,x):

    # Catch integer gamma values

    tol = 1E-5
    A = theta[0]
    y = np.log10(theta[1])
    nearest_int = np.around(y)
    dint = y - nearest_int
    if np.abs(dint) < tol:
        if dint >= 0:
            y = nearest_int+tol
        elif dint < 0:
            y = nearest_int-tol
            print(y)

    f_L = 1/(100*(x[-1]-x[0]))
    tau = 2*np.pi*np.abs(np.subtract.outer(x,x))

    scaling = A**2 * f_L ** (1-y)
    first_term = gamma(1-y)*np.sin(np.pi*y/2)*(f_L*tau)**(y-1)
    second_term = 1/(1-y) - (f_L*tau)**2 / (6-2*y)

    return scaling*(first_term-second_term)


def gwb(theta,x):

    y = 7/3
    A = theta[-3]

    f_L = 1/(100*(x[-1]-x[0]))
    tau = 2*np.pi*np.abs(np.subtract.outer(x,x))

    scaling = A**2 * f_L ** (1-y)
    first_term = gamma(1-y)*np.sin(np.pi*y/2)*(f_L*tau)**(y-1)
    second_term = 1/(1-y) - (f_L*tau)**2 / (6-2*y)

    return scaling*(first_term-second_term)

def rbf_gwb(theta,x):

    return rbf(theta,x) + gwb(theta, x)

def local_periodic_gwb(theta, x):

    return local_periodic(theta, x) + gwb(theta, x)

def matern_gwb(theta, x):

    return matern(theta, x) + gwb(theta, x)

def power_law_gwb(theta, x):

    return power_law(theta, x) + gwb(theta, x)

def GP(kernel, theta, data, mu_prior=[], sigma=[]):

    # Define test points (XT) and training data (X, y)
    XT, X, y = data
    n = len(X)
    
    # Calculate cov. matrix for join distribution
    K = kernel(10**theta, np.concatenate((X, XT)))
    
    # For non-noisy training data set sigma = 0
    if len(sigma)==0:
        sigma = np.zeros(n)
    if len(mu_prior)==0:
        mu_prior = np.zeros(n)

    # Sub-matrices of joint distribution, using cholesky decomp. for inversion
    K_XTX = K[n:,:n]
    K_XX = K[:n,:n]+np.diag(sigma**2)
    try:
    	ch_K_XX = cholesky(K_XX, lower=True)
    	K_XX_inv = inv(ch_K_XX.T) @ inv(ch_K_XX)
    except:
        K_XX_inv = inv(K_XX)

    K_XXT = K[:n,n:]
    K_XTXT = K[n:,n:]
    
    # Find conditioned mean function and covariance matrix
    m = K_XTX @ K_XX_inv @ (y-mu_prior)
    K = K_XTXT - K_XTX @ K_XX_inv @ K_XXT
    
    return (m, np.sqrt(np.diag(K)))