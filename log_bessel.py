import numpy as np
import scipy as sp
from scipy.special import loggamma as lg
from scipy.special import kv
from scipy.integrate import quad, quadrature

###########################################
#         APPROXIMATION FORMULAE          #
###########################################


# SciPy modified bessel function of the 
# second kind, valid in regimes where
# z < 200 and v < 700.

def scipy_bessel(v, z):

	res = np.log(kv(v,z))

	return res


# Rothwell approach for small z, based on
# Eq. 26 of Rothwell: Computation of the
# logarithm of Bessel functions of complex
# argument and fractional order
# https://onlinelibrary.wiley.com/doi/abs/
# 10.1002/cnm.972

def rothwell_lead(v, z):

    lead = 0.5*np.log(np.pi)-lg(v+0.5)-v*np.log(2*z)-z

    i_inf = np.isinf(lead)
    
    lead[i_inf] = -z[i_inf] + 0.5 * np.log(0.5*np.pi / z[i_inf])

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
    
    integral = np.zeros(np.shape(z))
    
    for i, z_tmp in enumerate(z):
        
        integral[i] = quad(inner_integral_rothwell,0,1,args=(v, z_tmp))[0]
        
    return np.log(integral)

def rothwell(v, z):

	z_shape = np.shape(z)
	z_flat = z.flatten()

	lead = rothwell_lead(v, z_flat)
	log_integral = compute_log_integral(v, z_flat)

	res = lead + log_integral
	res = np.reshape(res, z_shape)

	return res

# Asymptotic expansion at large v compared to z,
# Eq. 1.10 of Temme, Journal of Computational
# Physic, vol 19, 324 (1975)
# https://doi.org/10.1016/0021-9991(75)90082-0

def asymptotic_large_v(v, z):

	res = lg(v) - np.log(2) + v * (np.log(2)-np.log(z))

	return res

# Asymptotic expansion at large z compared to v,
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

    res = base + np.log(series_sum)

    return res


# Gamma integral-like formulation, with
# maximum moved outside of integral.
# TO DO: Need to find source for this 
# representation, believe it to be a
# reformulation of Rothwell Eq. 24.

def log_gamma_integral_func(t, v, z, log_offset=0):

	return (v - 1) * np.log(t) - 0.5 * z * (t + 1/t) - log_offset

def log_gamma_integral_max_t(v, z):

	return (np.sqrt(v**2 - 2*v + z**2 + 1)+v-1)/z

def gamma_integral(v, z):

	v_tmp = np.longdouble(v)
	z_tmp = np.longdouble(z)

	t_max = log_gamma_integral_max_t(v_tmp, z_tmp)
	value_at_max = log_gamma_integral_func(t_max, v_tmp, z_tmp)

	def f(t, v, z, val_max):

		log_value = log_gamma_integral_func(t, v, z, val_max)

		return 0.5 * np.exp(log_value)

	integral = np.zeros(np.shape(v))

	for i, v_ in enumerate(v_tmp):

		integral[i] = quad(f,0,np.inf,args=(v_, z_tmp, value_at_max[i]))[0]

	return np.log(integral)+value_at_max

# Trapezoidal rule for cosh integral 
# form as defined in 
# https://arxiv.org/pdf/1209.1547.pdf

def logsumexp(x):

	a = np.max(x)

	return a + np.log(np.sum(np.exp(x-a)))

def logcosh(x):

	return x + np.log1p(np.exp(-2*x)) - np.log(2)

def log_cosh_integral(t, v, z):

	return logcosh(v*t) - z * np.cosh(t)

def trap_cosh(v,z):

	z_shape = np.shape(z)
	z_flat = z.flatten()

	approx_max = np.arcsinh(v/z_flat)
	max_terms = 500
	h = approx_max / (0.5*max_terms)
	n = np.arange(0, max_terms, 1)

	res = np.zeros(np.shape(z_flat))

	for i in range(len(z_flat)):

		nh = n*h[i]
		terms = log_cosh_integral(nh, v, z_flat[i])
		res[i] = logsumexp(terms) + np.log(h[i])

	res = np.reshape(res, z_shape)

	return res

###########################################
#            METHOD SELECTION             #
###########################################

def rothwell_log_z_boundary(v):
    
    rothwell_max_log_z_over_v = 300

    return rothwell_max_log_z_over_v / (v-0.5)-np.log(2)

def method_indices(v, z):

	scipy_max_z = 695
	scipy_slope = 0.1
	scipy_intercept = np.log(127)

	asymp_v_slope = 1
	asymp_v_intercept = 8

	asymp_z_slope = 1
	asymp_z_intercept = -3
	asymp_z_log_min_v = 10

	rothwell_max_v = 50
	rothwell_max_z = 100000
	rothwell_max_log_z_over_v = 300

	trapezoid_min_v = 100

	scipy_1 = np.log(v) < (scipy_intercept + scipy_slope * np.log(z))
	scipy_2 = z < scipy_max_z
	i_scipy = scipy_1 & scipy_2

	rothwell_1 = v < rothwell_max_v
	rothwell_2 = z < rothwell_max_z
	rothwell_3 = (v <= 0.5) | (np.log(z) < rothwell_log_z_boundary(z))
	i_rothwell = rothwell_1 & rothwell_2 & rothwell_3 & ~i_scipy   #& ~i_gamma & ~i_rothwell_low & ~i_asymp_v_low

	trap_1 = np.log(v) < (asymp_v_slope * np.log(z) + asymp_v_intercept)
	trap_2 = np.log(v) > (asymp_z_slope * np.log(z) + asymp_z_intercept)
	trap_3 = (v > trapezoid_min_v) | (v > z)
	i_trap = trap_1 & trap_2 & trap_3 & ~i_rothwell & ~i_scipy     #& ~i_gamma & ~i_rothwell_low & ~i_asymp_v_low

	asymp_v_1 = v > z
	i_asymp_v = asymp_v_1 & ~i_trap & ~i_rothwell & ~i_scipy      #& ~i_gamma &  ~i_rothwell_low & ~i_asymp_v_low

	i_asymp_z = ~i_asymp_v & ~i_trap & ~i_rothwell & ~i_scipy #& ~i_gamma & ~i_rothwell_low & ~i_asymp_v_low


	return i_scipy, i_rothwell, i_trap, i_asymp_v, i_asymp_z

###########################################
#           TOP LEVEL FUNCTION            #
###########################################

def log_bessel_k(v, z):
    
    v = np.abs(v)
    res = np.zeros(np.shape(z))
    methods = [scipy_bessel, rothwell, trap_cosh,
    		   asymptotic_large_v, asymptotic_large_z]

    indeces = method_indices(v, z)
    
    for method, index in zip(methods, indeces):

        res[index] = method(v, z[index])

    return res





