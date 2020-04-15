import numpy as np
import scipy as sp
import gaussian_process as gp
from scipy.stats import uniform
import emcee as em
from schwimmbad import MPIPool
import sys
import os

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

def rbf_samples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 0))#np.log10(3*np.std(data[2], ddof=1))))
    efac_min, efac_max = (np.log10(np.min(data[5])), 1)
    equad_min, equad_max = (-8, 0)#np.log10(3*np.std(data[2], ddof=1)))
    
    r = uniform.rvs(size=(Nens, 4))
    
    return np.vstack((uniform.ppf(r[:,0], sigma_min, sigma_max-sigma_min), uniform.ppf(r[:,1], p_min, p_max-p_min),
    				  uniform.ppf(r[:,2], efac_min, efac_max-efac_min), uniform.ppf(r[:,3], equad_min, equad_max-equad_min))).T

def local_periodic_samples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 0))#np.log10(3*np.std(data[2], ddof=1))))
    efac_min, efac_max = (np.log10(np.min(data[5])), 1)
    equad_min, equad_max = (-8, 0)#np.log10(3*np.std(data[2], ddof=1)))
    
    r = uniform.rvs(size=(Nens, 5))
    
    return np.vstack((uniform.ppf(r[:,0], sigma_min, sigma_max-sigma_min), uniform.ppf(r[:,1], p_min, p_max-p_min), 
                      uniform.ppf(r[:,2], p_min, p_max-p_min), uniform.ppf(r[:,3], efac_min, efac_max-efac_min),
                      uniform.ppf(r[:,4], equad_min, equad_max-equad_min))).T

def matern_samples(Nens, data):
    
    diff = np.diff(data[1])
    p_min = np.log10(2*np.min(diff))
    p_max = np.log10(data[1][-1]-data[1][0])
    sigma_min, sigma_max = sorted((np.log10(np.min(data[5])), 0))#np.log10(3*np.std(data[2], ddof=1))))
    efac_min, efac_max = (np.log10(np.min(data[5])), 1)
    equad_min, equad_max = (-8, 0)#np.log10(3*np.std(data[2], ddof=1)))
    
    r = uniform.rvs(size=(Nens, 5))
    
    return np.vstack((uniform.ppf(r[:,0], sigma_min, sigma_max), uniform.ppf(r[:,1], -2, 5), 
                      uniform.ppf(r[:,2], p_min, p_max-p_min), uniform.ppf(r[:,3], efac_min, efac_max-efac_min),
                      uniform.ppf(r[:,4], equad_min, equad_max-equad_min))).T


pulsar_name, kernel_name, nsamples = sys.argv[1:]

kernel_info = {'RBF': {'kernel': gp.rbf, 'samples': rbf_samples}, 
			   'Local_Periodic': {'kernel': gp.local_periodic, 'samples': local_periodic_samples}, 
			   'Matern': {'kernel': gp.matern, 'samples': matern_samples}}

data_name = 'blank'
for root, dirs, files in os.walk('./pulsar_data'):
    for file in files:
        if pulsar_name in file:
            data_name = file

pulsar = np.genfromtxt('./pulsar_data/%s' %(data_name), usecols=(0,5,6))

data = (pulsar[1::2,0],pulsar[::2,0],pulsar[::2,1],pulsar[1::2,1],pulsar[1::2,2],pulsar[::2,2])

logprior = kernel_info[kernel_name]['logprior']
kernel = kernel_info[kernel_name]['kernel']

n = int(nsamples)
theta_samples = kernel_info[kernel_name]['samples'](n, data)

def lnL_sample(n):
    
    lnL_tmp = loglikelihood(theta_samples[n], data, kernel=kernel)
    
    return lnL_tmp

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
        
    loglikelihood_vals = np.array(pool.map(Z_est, range(n)))

lnL_max = np.max(loglikelihood_vals)
Z_est = 1/n * np.sum( np.exp( loglikelihood_vals-lnL_max ) )
Z_sq_est = 1/n * np.sum( np.exp( 2 * ( loglikelihood_vals-lnL_max ) ) )

Z_err, i = round_sig(np.sqrt(Z_sq-Z**2)/(np.sqrt(n)*Z))
Z_val = np.around(np.log(Z)+lnL_max, i)

path = f'./pulsar_results/{pulsar_name}/{nsamples}'

if not os.path.isdir(path):
    os.mkdir(path)

np.save(path + f'/{kernel_name}.npy', np.array([Z_val, z_err]))