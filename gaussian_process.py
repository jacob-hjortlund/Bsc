import numpy as np
import log_bessel as lb
from scipy.linalg import cholesky, inv
from scipy.special import loggamma as lg

def round_sig(x, sig=1):
    """
    Rounds a float to the given number of significant
    figures.

    Parameters
    -------------
    x    : Float
           Float to be rounded
    sig  : Integer
           Number of sig. figures
    """

    i = sig-int(np.floor(np.log10(np.abs(x))))-1
    return np.around(x, i), i


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

    # Adding white noise as parameter

    sigma_updated = np.sqrt(sigma**2 + theta[-1]**2)

    # Sub-matrices of joint distribution, using cholesky decomp. for inversion
    K_XTX = K[n:,:n]
    K_XX = K[:n,:n]+np.diag(sigma_updated**2)
    try:
    	ch_K_XX = cholesky(K_XX, lower=True)
    except:
        np.save(path + f'{kernel_name}_failed_matrix.npy', K_XX)
        print(K_XX)
        print(np.count_nonzero(K_XX<=0))

    K_XX_inv = inv(ch_K_XX.T) @ inv(ch_K_XX)#inv(K[:n,:n]+np.diag(sigma**2))
    K_XXT = K[:n,n:]
    K_XTXT = K[n:,n:]
    
    # Find conditioned mean function and covariance matrix
    m = K_XTX @ K_XX_inv @ (y-mu_prior)
    K = K_XTXT - K_XTX @ K_XX_inv @ K_XXT
    
    return (m, np.sqrt(np.diag(K)))