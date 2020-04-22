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

mean = np.fromstring(mean, sep=',')
sigma = np.fromstring(sigma, sep=',')

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

kernel_info = {'RBF': {'ndims': 4, 'kernel': gp.rbf, 'logprior': rbf_logprior}, 
               'Local_Periodic': {'ndims': 5, 'kernel': gp.local_periodic, 'logprior': local_periodic_logprior}, 
               'Matern': {'ndims': 5, 'kernel': gp.matern, 'logprior': matern_logprior}}

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