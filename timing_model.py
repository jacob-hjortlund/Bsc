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

pulsar_name, kernel_name, kernel_samples, nburnin, nsamples = sys.argv[1:]

path = f'./pulsar_results/{pulsar_name}/{kernel_samples}'
data_name = 'blank'
for root, dirs, files in os.walk('./pulsar_data'):
    for file in files:
        if pulsar_name in file:
            data_name = file

pulsar = np.genfromtxt('./pulsar_data/%s' %(data_name), usecols=(0,5,6))
x = pulsar[:,0]
y = pulsar[:,1]
sigma = pulsar[:,2]

gp_samples = np.load(path+f'/{kernel_name}_samples.npy')

def timing_model(x):

	p = 2*np.pi / 365.25

	return np.array([np.ones(len(x)), x, x**2, np.sin(p*x), np.cos(p*x)]).T

M = timing_model(x)

#S_inv = np.load(path+f'/{kernel_name}_S_inv.npy')
#Chi = np.load(path+f'/{kernel_name}_Chi.npy')
#ln_det_S = np.load(path+f'/{kernel_name}_log_det_S.npy')

ndims = 5
N = len(gp_samples)

kernel_info = {'RBF':  gp.rbf, 
			   'Local_Periodic': gp.local_periodic, 
			   'Matern': gp.matern,
			   'PL': gp.power_law}

kernel = kernel_info[kernel_name]

# Precalc necessary values

S_inv = np.empty((N, ndims, ndims))
Chi = np.empty((N, ndims))
ln_det_S = np.empty(N)

for i, sample in enumerate(gp_samples):

	# Update errors
	efac = 10**sample[-2]
	equad = 10**sample[-1]

	variance = (efac*sigma)**2 + equad**2

	# Calc covariance
	C = kernel(10**sample[:-2], x) + np.diag(variance)

	C_L = cholesky(C, lower=True, check_finite=False)
	CLM = solve_triangular(C_L, M, lower=True, check_finite=False)
	S_inv[i] = CLM.T @ CLM

	ln_det_S[i] = -np.linalg.slogdet(S_inv[i])[1]

	Chi[i] = inv(S_inv[i]) @ M.T @ solve(C, y)

################################
#       DEFINE FUNCTIONS       #
################################

A_min = -10
A_max = 2

def amplitude_inisamples(Nens):

	return uniform.rvs(A_min, A_max-A_min, size=(Nens, ndims))

def logposterior(theta):

	global S_inv
	global Chi
	global ln_det_S
	global ndims
	global N
	global A_min
	global A_max

	if not (np.all(theta <= A_max) and np.all(theta >= A_min)):

		return -np.inf

	D = np.tile(10**theta, (N,1))-Chi

	# Norm
	normalisation = -np.log(N) - 0.5*ndims * np.log(2*np.pi)

	# Exponent
	exponent = -0.5*(np.einsum('in,nmi,nm->n', D.T, S_inv, D)+ln_det_S)
	exponent_max = np.max(exponent)
	exponent_rescaled = exponent-exponent_max
	log_sum_exp = exponent_max + np.log(np.sum(np.exp(exponent_rescaled)))

	return normalisation+log_sum_exp

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

