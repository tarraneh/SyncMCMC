
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import emcee
import corner


"""
   Run a Markov-Chain Monte Carlo sampler to determine best fit parameters for a synchrotron model. 
   
   Fit produces F_v, v_a, v_m 
   
   F_v               ::  Flux normalization (F_v(v=v_a))
   v_a               ::  Self-absorption frequency
   v_m               ::  Peak frequency
   
   Known Parameters:
   
   z                 ::  Source redshift
   p                 ::  Power law index of electron Lorentz factor distribution
   epsilon_e         ::  Fraction of energy in electrons as a function of p
   epsilon_b         ::  Fraction of energy in the magnetic field
   d                 ::  Luminosity distance
   beta_1            ::  Spectral slope below break 1
   beta_2            ::  Spectral slope below break 2
   beta_3            ::  Spectral slope below break 3
   s_1               ::  Shape of spectrum at break 1
   s_2               ::  Shape of spectrum at break 2
   s_3               ::  Shape of spectrum at break 3
"""


# Define known parameters

p = 2.5
epsilon_e = 0.1 * (p-2.)/(p-1.)
epsilon_b = 0.1

beta_1 = 2.
beta_2 = 1./3.
beta_3 = (1.-p)/2.
beta5_1 = 5.0/2.0
beta5_2 = (1.0 - p)/2.0

s_1 = 1.5
s_2 = (1.76 + 0.05*p)
s_3 = (0.8 - 0.03*p)
# k = 2 (wind model)
s_4 = 3.63 * p - 1.60
s_5 = 1.25 - 0.18 * p


#### Synchrotron Models ####

# Define synchrotron spectrum for model 1
def spectrum(v,F_v,v_a,v_m):
    return F_v * (((v/v_a)**(-s_1*beta_1) + (v/v_a)**(-s_1*beta_2))**(-1./s_1)) * ((1 + (v/v_m)**(s_2*(beta_2-beta_3)))**(-1./s_2))

# Define synchrotron spectrum for model 2
def spectrum_2(v,F_4,v_a_2,v_m_2):
    phi = (v/v_m_2)
    return F_4 * (((phi)**(2.)) * np.exp(- s_4 * phi**(2./3.)) + phi**(5./2.) ) * ((1 + (v/v_a_2)**(s_5*(beta5_1-beta5_2)))**(-1./s_5))
    
    
 
 
#### Likelihood Functions and Priors ####

# Log likelihood function

def lnlike(theta, v, y, yerr):
    F_v,v_a,v_m,lnf = theta
    model = spectrum(v,F_v,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# Log priors

def lnprior(theta):
    F_v,v_a,v_m,lnf = theta
    if (1. < F_v < 55.) and (10**(8.) < v_a < 10**(11.)) and (10**(9.) < v_m < 10**(13.)) and (-3 < lnf < -0.01):
    #if (10 < F_v < 70.) and (10**(8.) < v_a < 10**(11.)) and (10**(8.) < v_m < 10**(11.)):    
        return 0.0
    return -np.inf


# Log probability

def lnprob(theta, v, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, v, y, yerr)
    
    
#### Sample the probability distribution ####

# Define number of dimensions and number of walkers

ndim, nwalkers = 4, 100


# Define initial positions of walkers in phase space

frand = np.random.normal(loc=F_true,size=nwalkers,scale=0.1)
varand = np.random.normal(loc=va_true,size=nwalkers,scale=1.E3)
vmrand = np.random.normal(loc=vm_true,size=nwalkers,scale=1.E3)
yerrand = np.random.normal(loc=-0.7,size=nwalkers,scale=0.1)

pos = np.column_stack((frand,varand,vmrand,yerrand)) 


# Run MCMC sampler

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(v3, flux3, err3))
sams = sampler.run_mcmc(pos, 1000)

# Burn off initial steps
samples = sampler.chain[:, 500:, :].reshape((-1, ndim))

samples[:, 2] = np.exp(samples[:, 2])
F_mcmc, va_mcmc, vm_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))


print F_mcmc
print va_mcmc
print vm_mcmc
