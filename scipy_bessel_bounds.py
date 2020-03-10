import numpy as np
import log_bessel as lb
from schwimmbad import MPIPool
from scipy.special import kv
import sys
import os

v_to_test = np.linspace(1E-7, 45000, 45000)
z_to_test = np.linspace(1E-7, 45500, 45000)

def function_test(v):
    
    return kv(v, z_to_test)

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit()
    
    res = np.array(pool.map(function_test, v_to_test))

path = './pulsar_results/bessel_test'

if not os.path.isdir(path):
	os.mkdir(path)

np.save(path + '/scipy_bessel_bounds_test.npy', res)