import numpy as np
import scipy as sp
from scipy.special import loggamma as lg
from scipy.integrate import quad, quad_vec, fixed_quad

###########################################
#         APPROXIMATION FORMULAE          #
###########################################


# Rothwell approach for small z, based on
# Eq. 26 of Rothwell: Computation of the
# logarithm of Bessel functions of complex
# argument and fractional order
# https://onlinelibrary.wiley.com/doi/abs/
# 10.1002/cnm.972

def rothwell_lead(v, z):

    lead = 0.5*np.log(np.pi)-lg(v+0.5)-v*np.log(2*z)-z
    
    lead[np.isinf(lead)] = -z + 0.5 * np.log(0.5*np.pi / z)

    return lead

def inner_integral_rothwell(u, v, z):

    n = 8
    v_mhalf = v - 0.5
    neg2v_m1 = -2. * v - 1.
    beta = (2. * n ) / (2. * v + 1.)

    uB = u**beta
    inv_u = -1. / u

    first = beta * np.exp(-uB) * (2 * z + uB)**v_mhalf * u**(n-1)
    second = np.exp(inv_u)
    if second > 0:
        second *= u**neg2v_m1
        if np.isinf(second):
            second = np.exp(inv_u+neg2v_m1*np.log(u))
        second *= (2. * z * u + 1)**v_mhalf

    return first + second

def compute_log_integral(v, z):
    
    integral = np.zeros(np.shape(v))
    
    for i, v_tmp in enumerate(v):
        
        integral[i] = quad(inner_integral_rothwell,0,1,args=(v_tmp, z))[0]
        
    return np.log(integral)

def rothwell(v, z):

	lead = rothwell_lead(v, z)
	log_integral = compute_log_integral(v, z)

	return lead + log_integral

# Asymptotic expansion at large v, Eq. 1.10
# of Temme, Journal of Computational
# Physic, vol 19, 324 (1975)
# https://doi.org/10.1016/0021-9991(75)90082-0

def asymptotic_large_v(v, z):

	return lg(v) - np.log(2) + v * (np.log(2)-np.log(z))

# Asymptotic expansion at large z,
# Eq. 10.40.2 of https://dlmf.nist.gov/10.40

def asymptotic_large_z(v, z):

    log_z = np.log(z)
    base = 0.5 * (np.log(np.pi) - np.log(2) - log_z) - z

    max_terms = 50
    v_squared_4 = v * v * 4
    a_k_z_k = 1
    series_sum = 1

    for k in range(1, max_terms+1):
        a_k_z_k *= (v_squared_4-(2 * k -1)**2) / (k * z * 8)
        series_sum += a_k_z_k

    return base + np.log(series_sum)

# Trapezoidal rule for integral form as
# defined in 
# https://arxiv.org/pdf/1209.1547.pdf


###########################################
#            METHOD SELECTION             #
###########################################

def rothwell_log_z_boundary(v):
    
    rothwell_max_log_z_over_v = 300

    return rothwell_max_log_z_over_v / (v-0.5)-np.log(2)

def method_indices(v, z):

    rothwell_max_v = 50
    rothwell_max_z = 100000
    rothwell_max_log_z_over_v = 300

    rothwell_1 = v < rothwell_max_v
    rothwell_2 = np.log(z) < rothwell_log_z_boundary(v)

    i_rothwell = np.logical_and(rothwell_1, rothwell_2)

    i_asymp_v = np.logical_and(v > z, ~i_rothwell)
    
    i_asymp_z = np.logical_and(np.logical_and(v > 10, ~i_rothwell),
                               ~i_asymp_v)
    

    return i_rothwell, i_asymp_z, i_asymp_v

###########################################
#           TOP LEVEL FUNCTION            #
###########################################

def log_bessel_k(v, z):
    
    v = np.abs(v)
    res = np.zeros(np.shape(v))
    methods = [rothwell, asymptotic_large_z, asymptotic_large_v]

    indeces = method_indices(v, z)
    
    for method, index in zip(methods, indeces):

        res[index] = method(v[index], z)

    return res, indeces





