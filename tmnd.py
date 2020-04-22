import numpy as np
import scipy as sp
import gaussian_process as gp
from scipy.stats import uniform
from scipy.stats import multivariate_normal
from scipy.linalg import inv
import emcee as em
from schwimmbad import MPIPool
import sys
import os

pulsar_name, kernel_name, nburnin, nsamples, mean, sigma = sys.argv[1:]

data_name = 'blank'
for root, dirs, files in os.walk('./pulsar_data'):
    for file in files:
        if pulsar_name in file:
            data_name = file

pulsar = np.genfromtxt('./pulsar_data/%s' %(data_name), usecols=(0,5,6))

data = (pulsar[1::2,0],pulsar[::2,0],pulsar[::2,1],pulsar[1::2,1],pulsar[1::2,2],pulsar[::2,2])

diff = np.diff(data[1])
p_min = np.log10(2*np.min(diff))
p_max = np.log10(data[1][-1]-data[1][0])
sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 1))#np.log10(3*np.std(data[2], ddof=1))))
efac_min, efac_max = np.log10(np.min(data[5])), 1
equad_min, equad_max = -8, np.log10(3*np.std(data[2], ddof=1))
v_min, v_max = -2, 3

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

def rbf_logprior(theta):
    
    s, l, efac, equad = theta.T

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_l = uniform.logpdf(l, p_min, p_max-p_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)
    
    return p_s + p_l + p_efac + p_equad

def local_periodic_logprior(theta):
    
    s, l, p, efac, equad = theta.T

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_l = uniform.logpdf(l, p_min, p_max-p_min)
    p_p = uniform.logpdf(p, p_min, p_max-p_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)
    
    return p_s + p_l + p_p + p_efac + p_equad 

def matern_logprior(theta):
    
    s, nu, l, efac, equad = theta.T

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_nu = uniform.logpdf(nu, -2, 5)
    p_l = uniform.logpdf(l, p_min, p_max-p_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)
    
    return p_s + p_nu + p_l + p_efac + p_equad 

def rbf_inisamples(Nens, data):
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens),
                      uniform.rvs(efac_min, efac_max-efac_min, size=Nens), uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

def local_periodic_inisamples(Nens, data):
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens),
                      uniform.rvs(p_min, p_max-p_min, size=Nens), uniform.rvs(efac_min, efac_max-efac_min, size=Nens),
                      uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

def matern_inisamples(Nens, data):
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(v_min, v_max-v_min, size=Nens),
                      uniform.rvs(p_min, p_max-p_min, size=Nens), uniform.rvs(efac_min, efac_max-efac_min, size=Nens),
                      uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

rbf_bounds = np.array([[sigma_min, sigma_max],
					   [p_min, p_max],
					   [efac_min, efac_max],
					   [equad_min, equad_max]])

local_per_bounds = np.array([[sigma_min, sigma_max],
							 [p_min, p_max],
							 [p_min, p_max],
							 [efac_min, efac_max],
							 [equad_min, equad_max]])

matern_bounds = np.array([[sigma_min, sigma_max],
						  [v_min, v_max],
						  [p_min, p_max],
						  [efac_min, efac_max],
						  [equad_min, equad_max]])

kernel_info = {'RBF': {'ndims': 4, 'kernel': gp.rbf, 'logprior': rbf_logprior, 'inisamples': rbf_inisamples, 'bounds': rbf_bounds}, 
			   'Local_Periodic': {'ndims': 5, 'kernel': gp.local_periodic, 'logprior': local_periodic_logprior, 'inisamples': local_periodic_inisamples, 'bounds': local_per_bounds}, 
			   'Matern': {'ndims': 5, 'kernel': gp.matern, 'logprior': matern_logprior, 'inisamples': matern_inisamples, 'bounds': matern_bounds}}

mean = np.fromstring(mean, sep=',')
sigma = np.fromstring(sigma, sep=',')
bounds = kernel_info[kernel_name]['bounds']

def lnprob_trunc_norm(x):

	global mean
	global sigma
	global bounds

	C = np.diag(sigma**2)

	if np.any(x < bounds[:,0]) or np.any(x > bounds[:,1]):
		return -np.inf
	else:
		return -0.5*(x-mean) @ (inv(C)) @ (x-mean)

def loglikelihood(theta, data, kernel=gp.rbf):
    
    """Data has structure (XT, X, y, yT, sigmaT, sigma)"""

    # Update errors
    efac = 10**theta[-2]
    equad = 10**theta[-1]

    sigmaT = np.sqrt( ( efac*data[4] )**2 + equad**2 )
    sigma = np.sqrt(  ( efac*data[5] )**2 + equad**2 )

    # normalisation
    norm = -0.5*len(data[0])*np.log(2*np.pi) - np.sum(np.log(sigmaT))

    # chi-squared
    chisq = np.sum(((data[3]-gp.GP(kernel, theta, data[:3], sigma=sigma)[0])/sigmaT)**2)

    return norm - 0.5*chisq

kernel = kernel_info[kernel_name]['kernel']

with MPIPool() as pool:
	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	Nens = 500   # number of ensemble points
	Nburnin = int(nburnin)   # number of burn-in samples
	Nsamples = int(nsamples)  # number of final posterior samples

	ndims = kernel_info[kernel_name]['ndims']

	np.random.seed()
	inisamples = kernel_info[kernel_name]['inisamples'](Nens, data) 
	# set up the sampler
	sampler = em.EnsembleSampler(Nens, ndims, lnprob_trunc_norm, pool=pool)#, args=argslist)
	sampler.run_mcmc(inisamples, Nsamples+Nburnin)

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

acl = sampler.get_autocorr_time(c=1, quiet=True)
print("The autocorrelation lengths are %s" %(acl))

theta_samples = sampler.chain[:, Nburnin::int(max(acl)), :].reshape((-1, ndims))
print("Number of independent samples is {}".format(len(theta_samples)))

path = f'./pulsar_results/{pulsar_name}/Bayes_factor/{nsamples}'

if not os.path.isdir(path):
    os.makedirs(path)

np.save(path + f'/{kernel_name}_samples.npy', theta_samples)