import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde, gamma, uniform, rv_continuous
from scipy.linalg import cholesky, LinAlgError, eigvalsh, eigh, inv
from scipy.special import gamma as gamma_func
from scipy.special import kv
import emcee as em
from schwimmbad import MPIPool
import sys
import os

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

    i = sig-int(np.floor(np.log10(abs(x))))-1
    return np.around(x, i), i


def rbf(theta,x):
    
    return theta[0]**2 * np.exp(-0.5 * theta[1]**-2 * np.subtract.outer(x,x)**2)

def periodic(theta,x):
    
    return theta[0]**2 * np.exp(-2 * theta[1]**-2 * np.sin(np.pi * np.subtract.outer(x,x) / theta[2])**2)

def local_periodic(theta,x):
    
    return rbf(theta,x) * periodic(theta,x)

def matern(theta, x):
    
    if theta[1] > 18:
        return rbf([theta[0], theta[2]], x)
    else:
        u = np.sqrt(2 * theta[1]) * abs(np.subtract.outer(x,x) / theta[2])
        u[u == 0.0] += np.finfo(float).eps
        K = theta[0] * 2**(1-theta[1]) / gamma_func(theta[1]) * (u)**theta[1] * kv(theta[1], u)

        return K

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

def loglikelihood(theta, data, kernel=rbf):
    
    """Data has structure (XT, X, y, yT, sigmaT, sigma)"""
    # Define global variables

    # normalisation
    norm = -0.5*len(data[0])*np.log(2*np.pi) - np.sum(np.log(data[4]))

    # chi-squared
    chisq = np.sum(((data[3]-GP(kernel, theta, data[:3], sigma=data[5])[0])/data[4])**2)

    return norm - 0.5*chisq

# class uniform_log_gen(rv_continuous):
#     def _pdf(self, x, a, b):
#         return uniform.pdf(10**x, a, b) * 10**x * np.log(10)
    
#     def _cdf(self, x, a, b):
#         return uniform.cdf(10**x, a, b)

# class gamma_log_gen(rv_continuous):
#     def _pdf(self, x, a, b):
#         return gamma.pdf(10**x, a, b) * 10**x * np.log(10)
    
#     def _cdf(self, x, a, b):
#         return gamma.cdf(10**x, a, b)

def rbf_logprior(theta, data):
    
    s, l = theta
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])#np.log10(2*np.max(diff))
    
    return uniform.logpdf(s, -2, 4) + uniform.logpdf(l, p_min, p_max-p_min)

def local_periodic_logprior(theta, data):
    
    s, l, p = theta
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])#np.log10(2*np.min(diff))
    
    return uniform.logpdf(s, -2, 4) + uniform.logpdf(l, p_min, p_max-p_min) + uniform.logpdf(p, p_min, p_max-p_min)

def matern_logprior(theta, data):
    
    s, nu, l = theta
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])#np.log10(2*np.max(diff))
    
    return uniform.logpdf(s, -2, 4) + uniform.logpdf(nu, -2, 4) + uniform.logpdf(l, p_min, p_max-p_min)

def rbf_inisamples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])#np.log10(2*np.max(diff))
    
    return np.vstack((uniform.rvs(-2,4, size=Nens),uniform.rvs(p_min, p_max-p_min, size=Nens))).T

def local_periodic_inisamples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))#np.log10(2*np.max(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    
    return np.vstack((uniform.rvs(-2,4, size=Nens),
                      uniform.rvs(p_min, p_max-p_min, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens))).T

def matern_inisamples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))#np.log10(2*np.max(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    
    return np.vstack((uniform.rvs(-2, 4, size=Nens),
    				  uniform.rvs(-2, 4, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens))).T

def rbf_samples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))#np.log10(2*np.max(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    
    r = uniform.rvs(size=(Nens, 2))
    
    return np.vstack((uniform.ppf(r[:,0], -2, 4), uniform.ppf(r[:,1], p_min, p_max-p_min))).T

def local_periodic_samples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))#np.log10(2*np.max(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    
    r = uniform.rvs(size=(Nens, 3))
    
    return np.vstack((uniform.ppf(r[:,0], -2, 4), uniform.ppf(r[:,1], p_min, p_max-p_min), 
                      uniform.ppf(r[:,2], p_min, p_max-p_min))).T

def matern_samples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))#np.log10(2*np.max(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    
    r = uniform.rvs(size=(Nens, 3))
    
    return np.vstack((uniform.ppf(r[:,0], -2, 4), uniform.ppf(r[:,1], -2, 4), 
                      uniform.ppf(r[:,2], p_min, p_max-p_min))).T

pulsar_name, kernel_name = sys.argv[1:]

kernel_info = {'RBF': {'ndims': 2, 'kernel': rbf, 'logprior': rbf_logprior, 'inisamples': rbf_inisamples, 'samples': rbf_samples}, 
			   'Local_Periodic': {'ndims': 3, 'kernel': local_periodic, 'logprior': local_periodic_logprior, 'inisamples': local_periodic_inisamples, 'samples': local_periodic_samples}, 
			   'Matern': {'ndims': 3, 'kernel': matern, 'logprior': matern_logprior, 'inisamples': matern_inisamples, 'samples': matern_samples}}

pulsar = np.loadtxt('./pulsar_data/%s.asc' %(pulsar_name), usecols=(0,1,2,7))

data = (pulsar[1::2,0],pulsar[::2,0],pulsar[::2,1],pulsar[1::2,1],pulsar[1::2,2],pulsar[::2,2])

logprior = kernel_info[kernel_name]['logprior']
kernel = kernel_info[kernel_name]['kernel']

def logposterior(theta): #data, logprior, kernel=rbf):
    
    global data
    global logprior
    global kernel

    lp = logprior(theta, data)
    
    if not np.isfinite(lp):
        
        return -np.inf
    
    return lp + loglikelihood(theta, data, kernel=kernel)

path = f'./pulsar_results/{pulsar_name}'

with MPIPool() as pool:
	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	Nens = 100   # number of ensemble points
	Nburnin = 3000   # number of burn-in samples
	Nsamples = 10000  # number of final posterior samples

	ndims = kernel_info[kernel_name]['ndims']

	np.random.seed()
	inisamples = kernel_info[kernel_name]['inisamples'](Nens, data) 
	# set up the sampler
	sampler = em.EnsembleSampler(Nens, ndims, logposterior, pool=pool)#, args=argslist)
	sampler.run_mcmc(inisamples, Nsamples+Nburnin)


acl = sampler.get_autocorr_time(c=1, quiet=True)
print("The autocorrelation lengths are %s" %(acl))

samples = sampler.chain[:, Nburnin::int(max(acl)), :].reshape((-1, ndims))
print("Number of independent samples is {}".format(len(samples)))

np.save(path + f'{kernel_name}.npy', samples)
