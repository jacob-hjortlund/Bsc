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

	if np.is_inf(lead):

		lead = -z + 0.5 * np.log(0.5*np.pi / z)

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
		if np.is_inf(second):
			second = np.exp(inv_u+neg2v_m1*np.log(u))
		second *= (2. * z * u + 1)**v_mhalf

	return first + second

def compute_log_integral(v, z):

	return np.log(quad(inner_integral_rothwell, 0, 1, args=(v, z))[0])

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
		if np.abs(a_k_z_k) < 1e-8:
			break

	return base + np.log(series_sum)




###########################################
#         CHOOSING AMONG FORMULAE         #
###########################################


def choose_computation(v, z):

	gamma_max_z = 200
	gamma_max_v = 3
	gamma_low_z = 0.01
	gamma_low_v = 0.001

	asymp_v_slope = 1
	asymp_v_intercept = 8

	asymp_z_slope = 1
	asymp_z_intercept = -3
	asymp_z_log_min_v = 10

	rothwell_max_v = 50
	rothwell_max_z = 100000
	rothwell_max_log_z_over_v = 300

	trapezoid_min_v = 100

	log_v, log_z = np.log(v), np.log(z)

	def rothwell_log_z_boundary(v):

		return rothwell_max_log_z_over_v / (v-0.5)-np.log(2)


	if v < gamma_low_v and z < gamma_low_z:

		return rothwell

	if v < gamma_max_v and z < gamma_max_z:

		return integral_gamma

	rothwell_check = v <= 0.5 or log_z < rothwell_log_z_boundary(v)
	if v < rothwell_max_z and z rothwell_max_z and rothwell_check:

		return rothwell

	trap_check1 = log_v < asymp_v_slope * log_z + asymp_v_intercept
	trap_check2 = log_v > asymp_z_slope * log_z + asymp_z_intercept
	trap_check3 = v > trapezoid_min_v or v > z

	if trap_check1 and trap_check2 and trap_check3:

		return trapezoid_cosh

	if v>z:

		return asymp_v

	if v > asymp_z_log_min_v:

		return asymp_z_log

	else return asymp_z

