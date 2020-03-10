import numpy as np
import log_bessel as lb
from schwimmbad import MPIPool
import sys

def logdiffexp(x1,x2):
    
    x1 = np.longdouble(x1)
    x2 = np.longdouble(x2)
    
    return x1+np.log1p(-np.exp(-(x1-x2)))

def forward_recursion(v,z):
    
    first, _ = lb.log_bessel_k(v+2,z)
    second = np.log(2) + np.log(np.longdouble(v+1)) - np.log(np.longdouble(z)) + lb.log_bessel_k(v+1,z)[0]
    indeces = lb.log_bessel_k(v+1,z)[1]
    
    return logdiffexp(first, second), indeces

def backward_recursion(v, z):
    
    first, indeces = lb.log_bessel_k(v-2, z)
    second = np.log(2)+np.log(v-1)-np.log(z)+lb.log_bessel_k(v-1, z)[0]
    
    return np.logaddexp(first, second), indeces

def recursion_test(v, z):
    
    rhs = np.zeros(np.shape(v))
    
    i_forward = v <= 1.
    i_backward = v > 1.
    
    rhs[i_forward], indices_forward = forward_recursion(v[i_forward], z)
    rhs[i_backward], indices_backward = backward_recursion(v[i_backward], z)
    
    lhs = lb.log_bessel_k(v, z)[0]

    return np.abs(lhs/rhs-1)

v_to_test = np.linspace(1,1E5,213029)
z_to_test = np.linspace(1E-7, 1E5, 213029)

def function_test(z):
    
    return recursion_test(v_to_test, z)

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit()
    
    res = np.array(pool.map(function_test, z_to_test))

path = './pulsar_results/bessel_test'

if not os.path.isdir(path):
	os.mkdir(path)

np.save(path + '/log_bessel_test.npy', res)