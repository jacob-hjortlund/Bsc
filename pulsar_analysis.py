import numpy as np
import scipy as sp
import gaussian_process as gp
from scipy.stats import uniform
import emcee as em
from schwimmbad import MPIPool
import sys
import os

def loglikelihood(theta, data, kernel=rbf):
    
    """Data has structure (XT, X, y, yT, sigmaT, sigma)"""
    # Define global variables

    # normalisation
    norm = -0.5*len(data[0])*np.log(2*np.pi) - np.sum(np.log(data[4]))

    # chi-squared
    chisq = np.sum(((data[3]-gp.GP(kernel, theta, data[:3], sigma=data[5])[0])/data[4])**2)

    return norm - 0.5*chisq

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

kernel_info = {'RBF': {'ndims': 2, 'kernel': gp.rbf, 'logprior': rbf_logprior, 'inisamples': rbf_inisamples, 'samples': rbf_samples}, 
			   'Local_Periodic': {'ndims': 3, 'kernel': gp.local_periodic, 'logprior': local_periodic_logprior, 'inisamples': local_periodic_inisamples, 'samples': local_periodic_samples}, 
			   'Matern': {'ndims': 3, 'kernel': gp.matern, 'logprior': matern_logprior, 'inisamples': matern_inisamples, 'samples': matern_samples}}

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
