import numpy as np
import scipy as sp
import gaussian_process as gp
from scipy.stats import uniform
from scipy.special import gamma
from scipy.misc import factorial
from scipy.linalg import cholesky, inv
import emcee as em
from schwimmbad import MPIPool
import sys
import os

def pulsar_timing_model(t,a,b,c,d,f):

	return a**2 * t + b*t + c + d*np.sin(f*t)

def power_law_sum(tau, f_L, y, n):

	denomenator = (-1)**n * (f_L*tau)**(2*n)
	numerator = factorial(2*n) * (2*n+1-y)  

	return  denomenator/numerator

def power_law_covariance(t, A, y):

	n = 1000
	f_L = 5 * 1/(t[-1]-t[0])
	tau = 2*np.pi*np.subtract.outer(t,t)
	scaling = A**2 / (f_L)**(y-1)
	first_term = gamma(1-y)*np.sin(np.pi*y/2)*(f_L*tau)**(y-1)

def rezip(x1, x2):

	n = len(x1)+len(x2)
	x = np.empty(n)
	x[::2] = x1
	x[1::2] = x2

	return x


def loglikelihood(theta, data, power_law=False, kernel=gp.rbf):
    
    """Data has structure (XT, X, y, yT, sigmaT, sigma)"""

    # Update errors
    efac = 10**theta[-2]
    equad = 10**theta[-1]

    sigmaT = np.sqrt( ( efac*data[4] )**2 + equad**2 )
    sigma = np.sqrt(  ( efac*data[5] )**2 + equad**2 )

    if not power_law:

    	# Calculate timing model
    	timing_model = pulsar_timing_model(data[0], *theta[:5])

    	# normalisation
    	norm = -0.5*len(data[0])*np.log(2*np.pi) - np.sum(np.log(sigmaT))

    	# chi-squared
    	gp_mean = gp.GP(kernel, theta[5:], data[:3], sigma=sigma)[0]
    	chisq = np.sum(((data[3]-gp_mean-timing_model)/sigmaT)**2)

    	return norm - 0.5*chisq
    
    else:

    	x = rezip(data[1], data[0])
    	y = rezip(data[2], data[3])
    	sigma = rezip(sigma, sigmaT)

    	cov = power_law_covariance(x, *theta[5:]) + np.diag(sigma**2)
    	ch_lower = cholesky(cov, lower=True)
    	det = np.prod(np.diag(ch_lower))**2
    	cov_inv = inv(ch_lower.T) @ inv(ch_lower)

    	# calculate timing model
    	timing_model = pulsar_timing_model(x, *theta[:5])

    	# normalisation
    	norm = -0.5*len(x)*np.log(2*np.pi) - 0.5*det

    	# chi-squared
    	chisq = (y-timing_model) @ cov_inv @ (y-timing_model)

    	return norm - 0.5*chisq


def rbf_logprior(theta, data):
    
    s, l, efac, equad = theta
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 1))#np.log10(3*np.std(data[2], ddof=1))))
    efac_min, efac_max = (np.log10(np.min(data[5])), 1)
    equad_min, equad_max = (-8, np.log10(3*np.std(data[2], ddof=1)))

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_l = uniform.logpdf(l, p_min, p_max-p_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)
    
    return p_s + p_l + p_efac + p_equad

def local_periodic_logprior(theta, data):
    
    s, l, p, efac, equad = theta
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 1))#np.log10(3*np.std(data[2], ddof=1))))
    efac_min, efac_max = (np.log10(np.min(data[5])), 1)
    equad_min, equad_max = (-8, np.log10(3*np.std(data[2], ddof=1)))

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_l = uniform.logpdf(l, p_min, p_max-p_min)
    p_p = uniform.logpdf(p, p_min, p_max-p_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)
    
    return p_s + p_l + p_p + p_efac + p_equad 

def matern_logprior(theta, data):
    
    s, nu, l, efac, equad = theta
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 1))#np.log10(3*np.std(data[2], ddof=1))))
    efac_min, efac_max = (np.log10(np.min(data[5])), 1)
    equad_min, equad_max = (-8, np.log10(3*np.std(data[2], ddof=1)))

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_nu = uniform.logpdf(nu, -2, 5)
    p_l = uniform.logpdf(l, p_min, p_max-p_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)
    
    return p_s + p_nu + p_l + p_efac + p_equad 

def rbf_inisamples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 1))#np.log10(3*np.std(data[2], ddof=1))))
    efac_min, efac_max = (np.log10(np.min(data[5])), 1)
    equad_min, equad_max = (-8, np.log10(3*np.std(data[2], ddof=1)))
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens),
                      uniform.rvs(efac_min, efac_max-efac_min, size=Nens), uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

def local_periodic_inisamples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 1))#np.log10(3*np.std(data[2], ddof=1))))
    efac_min, efac_max = (np.log10(np.min(data[5])), 1)
    equad_min, equad_max = (-8, np.log10(3*np.std(data[2], ddof=1)))
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens),
                      uniform.rvs(p_min, p_max-p_min, size=Nens), uniform.rvs(efac_min, efac_max-efac_min, size=Nens),
                      uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

def matern_inisamples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 1))#np.log10(3*np.std(data[2], ddof=1))))
    efac_min, efac_max = (np.log10(np.min(data[5])), 1)
    equad_min, equad_max = (-8, np.log10(3*np.std(data[2], ddof=1)))
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(-2, 5, size=Nens),
                      uniform.rvs(p_min, p_max-p_min, size=Nens), uniform.rvs(efac_min, efac_max-efac_min, size=Nens),
                      uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

pulsar_name, kernel_name, nburnin, nsamples = sys.argv[1:]

kernel_info = {'RBF': {'ndims': 4, 'kernel': gp.rbf, 'logprior': rbf_logprior, 'inisamples': rbf_inisamples}, 
			   'Local_Periodic': {'ndims': 5, 'kernel': gp.local_periodic, 'logprior': local_periodic_logprior, 'inisamples': local_periodic_inisamples}, 
			   'Matern': {'ndims': 5, 'kernel': gp.matern, 'logprior': matern_logprior, 'inisamples': matern_inisamples}}

data_name = 'blank'
for root, dirs, files in os.walk('./pulsar_data'):
    for file in files:
        if pulsar_name in file:
            data_name = file

pulsar = np.genfromtxt('./pulsar_data/%s' %(data_name), usecols=(0,5,6))

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

path = f'./pulsar_results/{pulsar_name}/{nsamples}'

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
	sampler = em.EnsembleSampler(Nens, ndims, logposterior, pool=pool)#, args=argslist)
	sampler.run_mcmc(inisamples, Nsamples+Nburnin)



print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

acl = sampler.get_autocorr_time(c=1, quiet=True)
print("The autocorrelation lengths are %s" %(acl))

samples = sampler.chain[:, Nburnin::int(max(acl)), :].reshape((-1, ndims))
print("Number of independent samples is {}".format(len(samples)))

if not os.path.isdir(path):
    os.makedirs(path)

np.save(path + f'/{kernel_name}_samples.npy', samples)

np.save(path + f'/{kernel_name}_lnprob.npy', sampler.lnprobability)

