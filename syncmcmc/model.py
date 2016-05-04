import numpy as np

"""

   Models and likelihood functions are defined for four different synchrotron spectra.

      Model 1: va < vm
      Model 2: vm < va
      Model 3: Superposition of models 1 and 2 with weighting factors; flux normalizations equal
      Model 4: Superposition of models 1 and 2 with weighting factors; unique flux normalization factors


   Known Parameters:
   
   p                 ::  Power law index of electron Lorentz factor distribution
   epsilon_e         ::  Fraction of energy in electrons as a function of p
   epsilon_b         ::  Fraction of energy in the magnetic field
   beta_1            ::  Spectral slope below break 1
   beta_2            ::  Spectral slope below break 2
   beta_3            ::  Spectral slope below break 3
   s_1               ::  Shape of spectrum at break 1
   s_2               ::  Shape of spectrum at break 2
   s_3               ::  Shape of spectrum at break 3
"""





### Define known parameters ### 


p = 2.5 
epsilon_e = 0.1 * (p-2.)/(p-1.)
epsilon_b = 0.1


# Spectral slopes

beta_1 = 2.
beta_2 = 1./3.
beta_3 = (1.-p)/2.
beta5_1 = 5.0/2.0
beta5_2 = (1.0 - p)/2.0



# Shape of spectrum at each break

s_1 = 1.5
s_2 = (1.76 + 0.05*p)
s_3 = (0.8 - 0.03*p)
s_4 = 3.63 * p - 1.60
s_5 = 1.25 - 0.18 * p







### Synchrotron Models ###

# Define synchrotron spectrum for Model 1 

def spectrum(v,F_v,v_a,v_m):
    return F_v * (((v/v_a)**(-s_1*beta_1) + (v/v_a)**(-s_1*beta_2))**(-1./s_1)) * ((1 + (v/v_m)**(s_2*(beta_2-beta_3)))**(-1./s_2))


# Define synchrotron spectrum for Model 2 

def spectrum_2(v,F_v,v_a,v_m):
    phi = (v/v_m)
    return F_v * (((phi)**(2.)) * np.exp(- s_4 * phi**(2./3.)) + phi**(5./2.) ) * ((1 + (v/v_a)**(s_5*(beta5_1-beta5_2)))**(-1./s_5))


# Define synchrotron spectrum for Model 3

def weighted_spectrum(v,F_v,v_a,v_m):
    phi = (v/v_m)
    return ((v_m/v_a)**2. * (F_v * (((v/v_a)**(-s_1*beta_1) + (v/v_a)**(-s_1*beta_2))**(-1./s_1)) * ((1 + (v/v_m)**(s_2*(beta_2-beta_3)))**(-1./s_2))) + (v_a/v_m)**2. * F_v * ((((phi)**(2.)) * np.exp(- s_4 * phi**(2./3.)) + phi**(5./2.) ) * ((1 + (v/v_a)**(s_5*(beta5_1-beta5_2)))**(-1./s_5))))/((v_a/v_m)**2. + (v_m/v_a)**2.)


# Define synchrotron spectrum for Model 4

def comb_spectrum(v,F_v,F_2,v_a,v_m):
    phi = (v/v_m)
    return ((v_m/v_a)**2. * (F_v * (((v/v_a)**(-s_1*beta_1) + (v/v_a)**(-s_1*beta_2))**(-1./s_1)) * ((1 + (v/v_m)**(s_2*(beta_2-beta_3)))**(-1./s_2))) + (v_a/v_m)**2. * F_2 * ((((phi)**(2.)) * np.exp(- s_4 * phi**(2./3.)) + phi**(5./2.) ) * ((1 + (v/v_a)**(s_5*(beta5_1-beta5_2)))**(-1./s_5))))/((v_a/v_m)**2. + (v_m/v_a)**2.)





### Log likelihood functions ###
## Includes a factor for fractional amount by which variances are underestimated

# Log likelihood for Model 1

def lnlike(theta, v, y, yerr):
    F_v,v_a,v_m,lnf = theta
    model = spectrum(v,F_v,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


# Log likelihood for Model 2

def lnlike_spec2(theta, v, y, yerr):
    F_v,v_a,v_m,lnf = theta
    model = spectrum_2(v,F_v,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


# Log likelihood for Model 3

def lnlike_spec3(theta, v, y, yerr):
    F_v,v_a,v_m,lnf = theta
    model = weighted_spectrum(v,F_v,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


# Log likelihood for Model 4

def lnlike_spec4(theta, v, y, yerr):
    F_v,F_v2,v_a,v_m,lnf = theta
    model = comb_spectrum(v,F_v,F_v2,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


