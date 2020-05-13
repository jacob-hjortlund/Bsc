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

def power_law_logprior(theta):

    s, gamma, efac, equad = theta

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_gamma = uniform.logpdf(gamma, gamma_min, gamma_max-gamma_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)

    return p_s + p_gamma + p_efac + p_equad

def matern_logprior(theta):
    
    s, nu, l, efac, equad = theta.T

    p_s = uniform.logpdf(s, sigma_min, sigma_max-sigma_min)
    p_nu = uniform.logpdf(nu, -2, 5)
    p_l = uniform.logpdf(l, p_min, p_max-p_min)
    p_efac = uniform.logpdf(efac, efac_min, efac_max-efac_min)
    p_equad = uniform.logpdf(equad, equad_min, equad_max-equad_min)
    
    return p_s + p_nu + p_l + p_efac + p_equad

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

    return ln_L

kernel_info = {'RBF': {'ndims': 4, 'kernel': gp.rbf, 'logprior': rbf_logprior}, 
               'Local_Periodic': {'ndims': 5, 'kernel': gp.local_periodic, 'logprior': local_periodic_logprior}, 
               'Matern': {'ndims': 5, 'kernel': gp.matern, 'logprior': matern_logprior},
               'PL': {'ndims': 4, 'kernel': gp.power_law, 'logprior': power_law_logprior}}


path = f'./pulsar_results/{pulsar_name}/Bayes_factor/{nsamples}'

theta_samples = np.load(path + f'/{kernel_name}_samples.npy')
kernel = kernel_info[kernel_name]['kernel']

def lnL_sample(n):

    global theta_samples
    global data
    global kernel

    lnL_tmp = loglikelihood(theta_samples[n], data, kernel=kernel)

    return lnL_tmp

n = len(theta_samples)

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
        
    loglikelihood_vals = np.array(pool.map(lnL_sample, range(n)))

lnL_max = np.max(loglikelihood_vals)
print(np.shape(loglikelihood_vals))
logprior_val = kernel_info[kernel_name]['logprior'](theta_samples[0])

z = 1/n * np.sum( np.exp( loglikelihood_vals + logprior_val - lnL_max ) / multivariate_normal.pdf(theta_samples, mean, sigma**2))
z_sq = 1/n * np.sum( np.exp( 2*(loglikelihood_vals + logprior_val - lnL_max) ) / multivariate_normal.pdf(theta_samples, mean, sigma**2)**2 )

print(np.sqrt((z_sq-z**2) / (np.log(10)*n*z**2)))

log10_z_err, i = round_sig( np.sqrt((z_sq-z**2) / (np.log(10)*n*z**2)) )
log10_z = np.around( 1/np.log(10) * (np.log(z)+lnL_max), i ) 

if not os.path.isdir(path):
    os.makedirs(path)

np.save(path + f'/{kernel_name}_z.npy', np.array([log10_z, log10_z_err]))
np.save(path + f'/{kernel_name}_lnl_samples.npy', loglikelihood_vals)