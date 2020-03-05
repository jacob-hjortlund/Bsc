import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde, gamma
from scipy.linalg import cholesky, LinAlgError, eigvalsh, eigh, inv
from numpy.linalg import inv
import time
import sys
import os
import six

def f(x):
    """The function to predict."""
    return x * np.sin(x)

def rbf(theta,x):
    
    return theta[0]**2 * np.exp(-0.5 * theta[1]**-2 * np.subtract.outer(x,x)**2)

def GP(kernel, theta, data, mu_prior=[], sigma=[]):
    
    global K_XX_error
    
    # Define test points (XT) and training data (X, y)
    XT, X, y = data
    n = len(X)
    
    # Calculate cov. matrix for join distribution
    K = kernel(theta, np.concatenate((X, XT)))
    
    # For non-noisy training data set sigma = 0
    if len(sigma)==0:
        sigma = np.zeros(n)
    if len(mu_prior)==0:
        mu_prior = np.zeros(n)
    
    # Sub-matrices of joint distribution, using cholesky decomp. for inversion
    K_XTX = K[n:,:n]
    K_XX = K[:n,:n]+np.diag(sigma**2)
    try:
        ch_K_XX = cholesky(K_XX, lower=True)#+np.diag(np.ones(len(K_XX)))*1E-10, lower=True)
    except:
        display(theta)
        display(K_XX)
        display(eigvalsh(K_XX))
        K_XX_error = K_XX
        
    K_XX_inv = inv(ch_K_XX.T) @ inv(ch_K_XX)#inv(K[:n,:n]+np.diag(sigma**2))
    K_XXT = K[:n,n:]
    K_XTXT = K[n:,n:]
    
    # Find conditioned mean function and covariance matrix
    m = K_XTX @ K_XX_inv @ (y-mu_prior)
    K = K_XTXT - K_XTX @ K_XX_inv @ K_XXT
    
    return (m, np.sqrt(np.diag(K)))

X = np.arange(0,10,0.25)

dy= 0.5
y = f(X) + np.random.normal(scale=dy, size=len(X))
sigma = np.ones(len(y))*dy

def prior_transform(theta):

    return gamma.ppf(theta,5.5)

def loglikelihood(theta):
    
    # Define global variables
    global X,y,sigma
    data = (X[1::2],X[::2],y[::2])
    sigma_training = sigma[::2]
    y_test = y[1::2]
    sigma_test = sigma[1::2]

    # normalisation
    norm = -0.5*len(data[0])*np.log(2*np.pi) - np.sum(np.log(sigma_test))

    # chi-squared
    chisq = np.sum(((y_test-GP(rbf, theta, data, sigma=sigma_training)[0])/sigma_test)**2)

    return norm - 0.5*chisq, []

import resource
curlimit = resource.getrlimit(resource.RLIMIT_STACK)
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,resource.RLIM_INFINITY))

nlive = 8192   # number of live points
ndims = 3      # number of parameters
nderived = 0   # number of derived parameters (this is zero)
tol = 0.5      # stopping criterion
basedir = os.path.join(os.getcwd(), 'polychord')  # output base directory
if not os.path.isdir(basedir):
    os.makedirs(basedir)                          # create base directory
    os.makedirs(os.path.join(basedir, 'clusters'))
fileroot = 'xsin'                         # output file name
broot = os.path.join(basedir, fileroot)

# import PolyChord
import pypolychord as pypoly
from pypolychord.settings import PolyChordSettings  # class for passing setup information
#from pypolychord.priors import UniformPrior         # pre-defined class for a uniform prior

# setup run settings using the PolyChordSetting class
pargs = {'nlive': nlive,
         'precision_criterion': tol,
         'base_dir': basedir,
         'file_root': fileroot,
         'write_resume': False, # don't output a resume file
         'read_resume': False}  # don't read a resume file
settings = PolyChordSettings(ndims, nderived, **pargs)

t0 = time.time()
output = pypoly.run_polychord(loglikelihood, ndims, nderived, settings, prior_transform)
t1 = time.time()

timepolychord = (t1-t0)

print("Time taken to run 'PyPolyChord' is {} seconds".format(timepolychord))

# reset stack resource limit
resource.setrlimit(resource.RLIMIT_STACK, curlimit)

print(six.u('log(Z) = {} \u00B1 {}'.format(output.logZ, output.logZerr)))
