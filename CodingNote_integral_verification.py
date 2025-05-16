import numpy as np
import math
from scipy.integrate import quad

# Debug mode cannot work for version <= 3.8
# conda create -n base_new python=3.13
# use "conda search python" to see all available python versions 
# "conda env list" see all conda environments



# example for using function "quad"
inv_acosh = lambda x: 1/np.cosh(x) if abs(x) < 710.4 else 0.

def integrand(x, n, c, d):
    return x**n*(inv_acosh(c-d*x)**4)

n = 2
c = 4
d = 1
delta = 0.2
# I = quad(integrand, c/d + delta, np.inf, args=(n,c,d))
# print(I)

# Calculate the integral via series summation;
N = 50 # maximum taylor expansion order


# test for convergence of series expansion of 1/cosh^4(dx-c) in domain (c/d, \infty);
x0 = c/d + 0.2
cosh_val = 0.
ratio_cd = c/d
for k in range(0, N+1):
    factor = (-1)**k*(k+1)*(k+2)*(k+3)/6
    cosh_val = cosh_val + factor*math.exp(-(2*k+4)*(d*x0-c))
cosh_val = cosh_val*16

print(cosh_val)
print(inv_acosh(d*x0-c)**4)

# test for convergence of series expansion of 1/cosh(dx-c) in domain (c/d-\pi/2, c/d+\pi/2);
# computing expansion coefficients
N = 10 # maximum number taylor expansion terms for 1/cosh(dx-c)
A_array = np.array([1])
for k in range(1, N+1):
    A2k = 0.
    for kp in range(0, k):
        A2k = A2k + A_array[kp]*math.comb(2*k, 2*kp)
    A2k = -A2k
    A_array = np.append(A_array, A2k)
# computing series summation
cosh_val = 0.
x0 = c/d + delta
for k in range(0, N+1):
    cosh_val = cosh_val + A_array[k]/math.factorial(2*k)*(d*delta)**(2*k)

print(inv_acosh(d*x0-c))
print(cosh_val)

# test for convergence of series expansion of 1/cosh^4(dx-c) in domain (c/d-\pi/2, c/d+\pi/2);
A_array_prime = np.array([])
Sk_array = np.array([])
for k in range(0, N+1):
    A_array_prime = np.append(A_array_prime, A_array[k]/math.factorial(2*k))
    Sk = 0.
    for k1 in range(0, k+1):
        for k2 in range(0, k+1-k1):
            for k3 in range(0, k+1-k1-k2):
                k4 = k-k1-k2-k3
                Sk = Sk + A_array_prime[k1]*A_array_prime[k2]*A_array_prime[k3]*A_array_prime[k4]
    Sk_array = np.append(Sk_array, [Sk])
x0 = c/d + 0.2
cosh_val = 0.
for k in range(0, N+1):
    cosh_val = cosh_val + Sk_array[k]*(d*x0-c)**(2*k)

print(cosh_val)
print(inv_acosh(d*x0-c)**4)


# test for n=2; series representation of the integral
# domain [c/d-\delta, c/d+\delta]
# definition follows the above code
# calculate the integral series summation
##### due to the existence of the factorial terms, the maximum number taylor expansion terms N ###############
################################   cannot be too large (i.e., 20~25)   #######################################
int_val = 0.
print("domain (c/d-\pi/2, c/d+\pi/2)")
for k in range(0, N+1):
    val_k = 0.
    for l in range(0, 2*k+1):
        val_tmp = (c/d+delta)**(2+l+1) - (c/d-delta)**(2+l+1) # n=2 here
        val_k = val_k + math.comb(2*k, l)*d**l*(-c)**(2*k-l)*val_tmp/(2+l+1)
    int_val = int_val + Sk_array[k]*val_k
print(quad(integrand, c/d - delta, c/d + delta, args=(n,c,d)))
print(int_val)


# test for n=2; series representation of the integral
# domain [c/d+\delta, \infty)
int_val = 0.
ratio_cd = c/d + delta
for k in range(0, N+1):
    factor = (-1)**k*(k+1)*(k+2)*(k+3)/6
    inv_expt_val = 1/(2*k+4)/d
    integral = inv_expt_val*ratio_cd**2 + 2*inv_expt_val**2*ratio_cd + 2*inv_expt_val**3
    integral = integral*math.exp(-(2*k+4)*d*delta)
    int_val = int_val + factor*integral
int_val = int_val*16

print(quad(integrand, c/d + delta, np.inf, args=(n,c,d)))
print(int_val)

# domain [0, c/d-\delta]
int_val = 0.
ratio_cd = c/d - delta
for k in range(0, N+1):
    factor = (-1)**k*(k+1)*(k+2)*(k+3)/6
    inv_expt_val = 1/(2*k+4)/d
    integral1 = (inv_expt_val*ratio_cd**2 - 2*inv_expt_val**2*ratio_cd + 2*inv_expt_val**3)*math.exp(-(2*k+4)*d*delta)
    integral2 = 2*inv_expt_val**3*math.exp(-(2*k+4)*c)
    int_val = int_val + factor*(integral1 - integral2)
int_val = int_val*16

print(quad(integrand, 0., c/d - delta, args=(n,c,d)))
print(int_val)


# test for series representation of the integral \int_0^\infty tanh(a-x)*x^3/cosh^2(a-x)dx
a = 1
N = 200
def integrand2(x, a, n):
    return np.tanh(a-x)*x**n*(inv_acosh(a-x)**2)

val_sum_1 = 0.
val_sum_2 = 0.
for k in range(1, N+1):
    k2 = k*k
    val_sum_1 = val_sum_1 + (-1)**(k+1)/k2*math.exp(-2*k*a)
    val_sum_2 = val_sum_2 + (-1)**(k+1)/k2
int_val = 3/2*val_sum_1 - 3*val_sum_2 - 3*a**2

print(quad(integrand2, 0., np.inf, args=(a, 3)))
print(int_val)

def integrand3(x, n, c):
    return x**n*(inv_acosh(x-c)**2)

"""
a1 = 4.;
a2 = 6.;
a3 = 4.;
a4 = 1.;
int_integrand = 0.
ratio_cd = c/d
for k in range(0, N+1):
    int_integrand_k = 0.
    for k1 in range(0, k+1):
        for k2 in range(0, k-k1+1):
            for k3 in range(0, k-k1-k2+1):
                for k4 in range(0, k-k1-k2-k3+1):
                    factor_multinomial = math.factorial(k)/(math.factorial(k1)*math.factorial(k2)*math.factorial(k3)\
                        *math.factorial(k4))
                    coef_a = (a1**k1)*(a2**k2)*(a3**k3)*(a4**k4)
                    inv_expt_val = 1/((2*(k1+2*k2+3*k3+4*k4)+4)*d)
                    integral_term1 = inv_expt_val*ratio_cd**2 + 2*inv_expt_val**2*ratio_cd + 2*inv_expt_val**3
                    integral_term2 = inv_expt_val*ratio_cd**2 - 2*inv_expt_val**2*ratio_cd + 2*inv_expt_val**3\
                        - 2*inv_expt_val**3*math.exp(-(2*(k1+2*k2+3*k3+4*k4)+4)*c)
                    int_integrand_k = factor_multinomial*coef_a*(integral_term1 + integral_term2)
    int_integrand = int_integrand + (-1)**k*int_integrand_k

print(int_integrand)
"""