import numpy as np
import scipy as sp
import gaussian_process as gp
from scipy.stats import uniform
from scipy.special import gamma
from scipy.linalg import cholesky, inv, solve_triangular, svd, solve
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

	p = 2*np.pi / 365.25

	return np.array([np.ones(len(x)), x, x**2, np.sin(p*x), np.cos(p*x)]).T

# Set up data with expected structure

M = timing_model(x)
F, _, _ = svd(M)
G = F[:, len(M[0]):]
data = [pulsar[:,0], y, sigma, G.T @ y, G, M]

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

	""" Data has structure (x, y, sigma, G.T @ y, G, M) """

	# Update errors
	efac = 10**theta[-2]
	equad = 10**theta[-1]

	variance = (efac*data[2])**2 + equad**2

	# Calculate and update covariance matrix
	C = kernel(10**theta[:-2], data[0]) + np.diag(variance)
	GCG = data[4].T @ C @ data[4]

	# Decomp and determinant
	try:
		GCG_L = cholesky(GCG, lower=True, overwrite_a=True, check_finite=False)
	except:
		return -np.inf

	ln_det_GCG = np.sum(np.log(np.diag(GCG_L)))

	# Calulate likelihood
	normalisation = -0.5 * len(G[0]) * np.log(2*np.pi) - 0.5 * ln_det_GCG
	GCG_D = solve_triangular(GCG_L, data[3], lower=True, check_finite=False)
	ln_L =  normalisation - 0.5 * GCG_D @ GCG_D

	# Calculate meta values for marginalization

	C_L = cholesky(C, lower=True, overwrite_a=True, check_finite=False)
	CLM = solve_triangular(C_L, data[5], lower=True, check_finite=False)
	S_inv = CLM.T @ CLM

	ln_det_S = -np.linalg.slogdet(S_inv)[1]

	chi = inv(S_inv) @ data[5].T @ solve(C, data[1])

	return ln_L, ln_det_S, S_inv, chi

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

def rbf_inisamples(Nens):
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens),
                      uniform.rvs(efac_min, efac_max-efac_min, size=Nens), uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

def local_periodic_inisamples(Nens):
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens),
                      uniform.rvs(p_min, p_max-p_min, size=Nens), uniform.rvs(efac_min, efac_max-efac_min, size=Nens),
                      uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

def matern_inisamples(Nens):
    
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

    lnp = logprior(theta)
    
    if not np.isfinite(lnp):
        
        return -np.inf, np.nan, np.empty((5,5)), np.empty(5)

    lnL, ln_det_S, S_inv, chi = loglikelihood(theta, data, kernel=kernel)
    
    return lnp + lnL, ln_det_S, S_inv, chi

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
	inisamples = kernel_info[kernel_name]['inisamples'](Nens) 

	dtype = [("log_det_S", np.float64), ("S_inv", np.ndarray), ("Chi", np.ndarray)]

	sampler = em.EnsembleSampler(Nens, ndims, logposterior, pool=pool, blobs_dtype=dtype,
								 moves=[(em.moves.DEMove(), 0.8), (em.moves.DESnookerMove(), 0.2),])
	sampler.run_mcmc(inisamples, Nsamples+Nburnin)


#############################################
#       SAVE SAMPLES AND DISPLAY INFO       #
#############################################

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

acl = sampler.get_autocorr_time(c=1, quiet=True)
print("The autocorrelation lengths are %s" %(acl))

# Samples
max_acl = int(max(acl))
samples = sampler.chain[:, Nburnin::max_acl, :].reshape((-1, ndims))
ln_samples = sampler.lnprobability[:, 5000::max_acl].reshape(-1)
print("Number of independent samples is {}".format(len(samples)))

# Meta data
blobs = sampler.get_blobs()
ln_det_S = blobs['log_det_S'][500::max_acl,:].reshape(-1)
S_inv = np.stack(blobs['S_inv'][500::max_acl,:].reshape(-1))
Chi = np.stack(blobs['Chi'][500::max_acl,:].reshape(-1))

if not os.path.isdir(path):
    os.makedirs(path)

np.save(path + f'/{kernel_name}_samples.npy', samples)

np.save(path + f'/{kernel_name}_lnprob.npy', ln_samples)

np.save(path + f'/{kernel_name}_log_det_S.npy', ln_det_S)

np.save(path + f'/{kernel_name}_S_inv.npy', S_inv)

np.save(path + f'/{kernel_name}_Chi.npy', Chi)

