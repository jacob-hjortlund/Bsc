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

# Pre-calc prior bounds

diff = np.diff(x)
p_min = np.log10(2*np.min(diff))
p_max = np.log10(x[-1]-x[0])
sigma_min, sigma_max = sorted((np.log10(np.min(sigma)), 1))
efac_min, efac_max = (np.log10(np.min(sigma)), 1)
equad_min, equad_max = (-8, np.log10(3*np.std(y, ddof=1)))
nu_min, nu_max = (-2, 3)
gamma_min, gamma_max = (1,7)

# Load mean and sigma

load_path = f'./pulsar_results/{pulsar_name}/{nsamples}/{kernel_name}_samples.npy'
mcmc_samples = np.load(load_path)
percentiles =np.percentile(mcmc_samples, [16,50,84], axis=0)
sigma = np.max(np.diff(percentiles,axis=0),axis=0)
mean = percentiles[1,:]

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

def rbf_inisamples(Nens):
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens),
                      uniform.rvs(efac_min, efac_max-efac_min, size=Nens), uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

def local_periodic_inisamples(Nens):
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(p_min, p_max-p_min, size=Nens),
                      uniform.rvs(p_min, p_max-p_min, size=Nens), uniform.rvs(efac_min, efac_max-efac_min, size=Nens),
                      uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

def matern_inisamples(Nens):
    
    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(v_min, v_max-v_min, size=Nens),
                      uniform.rvs(p_min, p_max-p_min, size=Nens), uniform.rvs(efac_min, efac_max-efac_min, size=Nens),
                      uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T

def power_law_inisamples(Nens):

    return np.vstack((uniform.rvs(sigma_min, sigma_max-sigma_min, size=Nens), uniform.rvs(gamma_min, gamma_max-gamma_min, size=Nens),
                      uniform.rvs(efac_min, efac_max-efac_min, size=Nens), uniform.rvs(equad_min, equad_max-equad_min, size=Nens))).T


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

power_law_bounds = np.array([[sigma_min, sigma_max],
                            [gamma_min, gamma_max],
                            [efac_min, efac_max],
                            [equad_min, equad_max]])

kernel_info = {'RBF': {'ndims': 4, 'kernel': gp.rbf, 'inisamples': rbf_inisamples, 'bounds': rbf_bounds}, 
               'Local_Periodic': {'ndims': 5, 'kernel': gp.local_periodic, 'inisamples': local_periodic_inisamples, 'bounds': local_per_bounds}, 
               'Matern': {'ndims': 5, 'kernel': gp.matern, 'inisamples': matern_inisamples, 'bounds': matern_bounds},
               'PL': {'ndims': 4, 'kernel': gp.power_law, 'inisamples': power_law_inisamples, 'bounds': power_law_bounds}}

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

kernel = kernel_info[kernel_name]['kernel']

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