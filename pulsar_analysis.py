import numpy as np
import scipy as sp
import gaussian_process as gp
from scipy.stats import uniform
from scipy.special import gamma
from scipy.misc import factorial
from scipy.linalg import cholesky, inv, solve_triangular, svd
import emcee as em
from schwimmbad import MPIPool
import sys
import os

###############################
#            SETUP            #
###############################

# Load data

pulsar_name, kernel_name, nburnin, nsamples = sys.argv[1:]

data_name = 'blank'
for root, dirs, files in os.walk('./pulsar_data'):
    for file in files:
        if pulsar_name in file:
            data_name = file

pulsar = np.genfromtxt('./pulsar_data/%s' %(data_name), usecols=(0,5,6))
x = pulsar[:,0]
y = pulsar[:,1]
sigma = pulsar[:,2]

#data = (pulsar[1::2,0],pulsar[::2,0],pulsar[::2,1],pulsar[1::2,1],pulsar[1::2,2],pulsar[::2,2])

def timing_model(x):

	return np.array([np.ones(len(x)), x, x**2, np.sin(x), np.cos(x)]).T

# Set up data with expected structure

M = timing_model(pulsar[:,0])
F, _, _ = svd(M)
G = F[:, len(M[0]):]
data = [pulsar[:,0], G.T @ y, G, sigma]

# Pre-calc prior bounds

diff = np.diff(x)
p_min = np.log10(2*np.min(diff))
p_max = np.log10(x[-1]-x[0])
sigma_min, sigma_max = sorted((np.log10(np.min(sigma)), 1))#np.log10(3*np.std(data[2], ddof=1))))
efac_min, efac_max = (np.log10(np.min(sigma)), 1)
equad_min, equad_max = (-8, np.log10(3*np.std(y, ddof=1)))
nu_min, nu_max = (-2, 3)

################################
#       DEFINE FUNCTIONS       #
################################

def loglikelihood(theta, data, kernel=gp.rbf):

	""" Data has structure (x, G.T @ y, G, sigma) """

	# Update errors
	efac = 10**theta[-2]
	equad = 10**theta[-1]

	variance = (efac*data[3])**2 + equad**2

	# Calculate and update covariance matrix
	C = kernel(10**theta[:-2], data[0]) + np.diag(variance)
	GCG = data[2].T @ C @ data[2]

	# Decomp and determinant
	try:
		GCG_L = cholesky(GCG, lower=True, overwrite_a=True, check_finite=False)
	except:
		return -np.inf

	det_GCG = np.prod(np.diag(GCG_L))

	# Calulate likelihood
	normalisation = -0.5*len(G[0])*np.log(2*np.pi) - 0.5*np.log(det_GCG)
	ln_L =  normalisation - 0.5*np.norm(solve_triangular(GCG_L, data[1], lower=True, check_finite=False))

	return ln_L

def rbf_logprior(theta):
    
    s, l, efac, equad = theta

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_l = uniform.logpdf(l, p_min, p_max-p_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)
    
    return p_s + p_l + p_efac + p_equad

def local_periodic_logprior(theta):
    
    s, l, p, efac, equad = theta

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_l = uniform.logpdf(l, p_min, p_max-p_min)
    p_p = uniform.logpdf(p, p_min, p_max-p_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)
    
    return p_s + p_l + p_p + p_efac + p_equad 

def matern_logprior(theta):
    
    s, nu, l, efac, equad = theta

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_nu = uniform.logpdf(nu, nu_min, nu_max-nu_min)
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
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(-2, 5, size=Nens),
                      uniform.rvs(p_min, p_max-p_min, size=Nens), uniform.rvs(efac_min, efac_max-efac_min, size=Nens),
                      uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

kernel_info = {'RBF': {'ndims': 4, 'kernel': gp.rbf, 'logprior': rbf_logprior, 'inisamples': rbf_inisamples}, 
			   'Local_Periodic': {'ndims': 5, 'kernel': gp.local_periodic, 'logprior': local_periodic_logprior, 'inisamples': local_periodic_inisamples}, 
			   'Matern': {'ndims': 5, 'kernel': gp.matern, 'logprior': matern_logprior, 'inisamples': matern_inisamples}}

logprior = kernel_info[kernel_name]['logprior']
kernel = kernel_info[kernel_name]['kernel']

##################################
#       MCMC SETUP AND RUN       #
##################################

def logposterior(theta):
    
    # Import global values
    # to minimize MPI overhead

    global data
    global logprior
    global kernel

    lp = logprior(theta)
    
    if not np.isfinite(lp):
        
        return -np.inf
    
    return lp + loglikelihood(theta, data, kernel=kernel)

path = f'./pulsar_results/{pulsar_name}/{nsamples}'

with MPIPool() as pool:
	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	Nens = 50   # number of ensemble points
	Nburnin = int(nburnin)   # number of burn-in samples
	Nsamples = int(nsamples)  # number of final posterior samples

	ndims = kernel_info[kernel_name]['ndims']

	np.random.seed()
	inisamples = kernel_info[kernel_name]['inisamples'](Nens, data) 

	sampler = em.EnsembleSampler(Nens, ndims, logposterior, pool=pool)
	sampler.run_mcmc(inisamples, Nsamples+Nburnin)


#############################################
#       SAVE SAMPLES AND DISPLAY INFO       #
#############################################

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

acl = sampler.get_autocorr_time(c=1, quiet=True)
print("The autocorrelation lengths are %s" %(acl))

samples = sampler.chain[:, Nburnin::int(max(acl)), :].reshape((-1, ndims))
print("Number of independent samples is {}".format(len(samples)))

if not os.path.isdir(path):
    os.makedirs(path)

np.save(path + f'/{kernel_name}_samples.npy', samples)

np.save(path + f'/{kernel_name}_lnprob.npy', sampler.lnprobability)

