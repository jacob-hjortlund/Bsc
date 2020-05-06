import numpy as np
import scipy as sp
import gaussian_process as gp
from scipy.stats import uniform
from scipy.special import gamma
import emcee as em
from schwimmbad import MPIPool
import sys
import os

###############################
#            SETUP            #
###############################

# Load precalcs

pulsar_name, kernel_name, nburnin, nsamples = sys.argv[1:]

path = f'./pulsar_results/{pulsar_name}/{nsamples}'

S_inv = np.load(path+f'{kernel_name}_S_inv.npy')
Chi = np.load(path+f'{kernel_name}_Chi.npy')
ln_det_S = np.load(path+f'{kernel_name}_log_det_S.npy')

ndims = len(S_inv[0])
N = len(S_inv)

################################
#       DEFINE FUNCTIONS       #
################################

def amplitude_inisamples(Nens):

	return uniform.rvs(-2, 4, size=(Nens, ndims))

def logposterior(theta):

	global S_inv
	global Chi
	global ln_det_S
	global ndims
	global N

	D = np.tile(theta, (N,1))-Chi
	normalisation = -np.log(N) - 0.5*ndims * np.log(2*np.pi) - 0.5*np.sum(ln_det_S)
	exponent = 0.5*np.einsum('in,nmi,nm->', D.T, S_inv, D)

	return normalisation-exponent

##################################
#       MCMC SETUP AND RUN       #
##################################

with MPIPool() as pool:
	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	Nens = 50   # number of ensemble points
	Nburnin = int(nburnin)   # number of burn-in samples
	Nsamples = int(nsamples)  # number of final posterior samples

	ndims = ndims

	np.random.seed()
	inisamples = amplitude_inisamples(Nens) 

	sampler = em.EnsembleSampler(Nens, ndims, logposterior, pool=pool,
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

if not os.path.isdir(path):
    os.makedirs(path)

np.save(path + f'/{kernel_name}_timing_model_samples.npy', samples)

np.save(path + f'/{kernel_name}_timing_model_lnprob.npy', ln_samples)

