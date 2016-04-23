from priorclasses import FluxFrequencyPriorsSimpleModel,UniformPrior,FluxFrequencyPriorsSimpleModelCombinedSpectrum
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import emcee
import corner
import os
import argparse
sns.set_style("white")

"""
   Run a Markov-Chain Monte Carlo sampler to determine best fit parameters for a synchrotron model. 

   No correction for underestimate of variances.

   Usage: model.py [options] -i filename

   Outputs best fit parameters for F_v, v_a, and v_m and a plot of the fit.

   Options:

   -i     --input        Specify data file
   -r     --raw          Plot raw data
   -fp    --fprior       Speicify lower and upper bounds for prior on flux normalization factor (Default: 1:50)
   -vap   --vaprior      Specify lower and upper bounds for prior on self absorption frequency (Default 1E9:1E13)
   -vam   --vamprior     Specify lower and upper bounds for prior on characteristic frequency (Default 1E9:1E13)
   -t     --trace        Plot MCMC traces for F_v, v_a, v_m
   -c     --corner       Plot corner plots
   -F     --F_true       Specify true value for F_v
   -vat   --va_true      Specify true value for va
   -vmt   --vm_true      Specify true value for vm 

   
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


# Read in and process command line options

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Specify input file for retrieving data', dest='data',type=str,default='None',action='store',required=True)
parser.add_argument('-r', '--raw', help='Plot raw data', dest='raw',default='None',action='store_true',required=False)
parser.add_argument('-fp', '--fprior', help='Define uniform prior bounds for flux normalization, -fp lower upper', dest='fluxprior',default='1. 55.',action='store',required=False)
parser.add_argument('-vap', '--vaprior', help='Define uniform prior bounds for self absorption frequency, -vap lower upper', dest='vaprior',default='1E8 1E13',action='store',required=False)
parser.add_argument('-vmp', '--vmprior', help='Define uniform prior bounds for characteristic frequency, -vam lower upper', dest='vmprior',default='1E8 1E13',action='store',required=False)
parser.add_argument('-c', '--corner', help='Plot corner plots', dest='corner',default='None',action='store_true',required=False)
parser.add_argument('-t', '--trace', help='Plot MCMC traces', dest='traces',default='None',action='store_true',required=False)
parser.add_argument('-Ft', '--F_true', help='Specify true value for F_v', type=str,dest='F_true',action='store',required=True)
parser.add_argument('-vat', '--va_true', help='Specify true value for v_a', type=str,dest='va_true',action='store',required=True)
parser.add_argument('-vmt', '--vm_true', help='Specify true value for v_m', type=str,dest='vm_true',action='store',required=True)


args = parser.parse_args()

data_file = args.data
plot_raw_data = args.raw
flux_bounds = args.fluxprior
va_bounds = args.vaprior
vm_bounds = args.vmprior
plot_corner = args.corner
plot_traces = args.traces
F_true = (args.F_true)
va_true = (args.va_true)
vm_true = (args.vm_true)



# Load data

flux = []
freqs = []
error = []
for line in open(data_file):
   lines = line.strip()
   if not line.startswith("#"):
      columns = line.split(',')
      freqs.append(columns[0])
      flux.append(columns[1])
      error.append(columns[2].rstrip('\n'))

flux = np.array(flux).astype(float)
freqs = np.array(freqs).astype(float)
error = np.array(error).astype(float)



# Plot raw data if argument -r passed

if plot_raw_data is True:
   plt.figure()
   plt.scatter(freqs,flux)
   plt.xscale('log')
   plt.yscale('log')
   plt.show()



# Define synchrotron spectrum for model 1 in Granot and Sari

def spectrum(v,F_v,v_a,v_m):
    return F_v * (((v/v_a)**(-s_1*beta_1) + (v/v_a)**(-s_1*beta_2))**(-1./s_1)) * ((1 + (v/v_m)**(s_2*(beta_2-beta_3)))**(-1./s_2))


## Define synchrotron spectrum for model 2 in Granot and Sari

def spectrum_2(v,F_v,v_a,v_m):
    phi = (v/v_m)
    return F_v * (((phi)**(2.)) * np.exp(- s_4 * phi**(2./3.)) + phi**(5./2.) ) * ((1 + (v/v_a)**(s_5*(beta5_1-beta5_2)))**(-1./s_5))

## Define weighted synchrotron spectrum with flux normalization factors equal

def weighted_spectrum(v,F_v,v_a,v_m):
    phi = (v/v_m)
    return ((v_m/v_a)**2. * (F_v * (((v/v_a)**(-s_1*beta_1) + (v/v_a)**(-s_1*beta_2))**(-1./s_1)) * ((1 + (v/v_m)**(s_2*(beta_2-beta_3)))**(-1./s_2))) + (v_a/v_m)**2. * F_v * ((((phi)**(2.)) * np.exp(- s_4 * phi**(2./3.)) + phi**(5./2.) ) * ((1 + (v/v_a)**(s_5*(beta5_1-beta5_2)))**(-1./s_5))))/((v_a/v_m)**2. + (v_m/v_a)**2.)

## Define weighted synchrortron spectrum with unique flux normalization factors

def comb_spectrum(v,F_v,F_2,v_a,v_m):
    phi = (v/v_m)
    return ((v_m/v_a)**2. * (F_v * (((v/v_a)**(-s_1*beta_1) + (v/v_a)**(-s_1*beta_2))**(-1./s_1)) * ((1 + (v/v_m)**(s_2*(beta_2-beta_3)))**(-1./s_2))) + (v_a/v_m)**2. * F_2 * ((((phi)**(2.)) * np.exp(- s_4 * phi**(2./3.)) + phi**(5./2.) ) * ((1 + (v/v_a)**(s_5*(beta5_1-beta5_2)))**(-1./s_5))))/((v_a/v_m)**2. + (v_m/v_a)**2.)





# Log likelihood function

def lnlike(theta, v, y, yerr):
    F_v,v_a,v_m= theta
    model = spectrum(v,F_v,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def lnlike_spec2(theta, v, y, yerr):
    F_v,v_a,v_m = theta
    model = spectrum_2(v,F_v,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def lnlike_spec3(theta, v, y, yerr):
    F_v,v_a,v_m = theta
    model = weighted_spectrum(v,F_v,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnlike_spec4(theta, v, y, yerr):
    F_v,F_v2,v_a,v_m = theta
    model = comb_spectrum(v,F_v,F_v2,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))





# Define priors

priors = FluxFrequencyPriorsSimpleModel(UniformPrior(float(flux_bounds.split(' ')[0]),float(flux_bounds.split(' ')[1])),
          UniformPrior(float(va_bounds.split(' ')[0]),float(va_bounds.split(' ')[1])),
          UniformPrior(float(vm_bounds.split(' ')[0]),float(vm_bounds.split(' ')[1])))

priors_spec4 = FluxFrequencyPriorsSimpleModelCombinedSpectrum(UniformPrior(float(flux_bounds.split(' ')[0]),float(flux_bounds.split(' ')[1])),
          UniformPrior(float(flux_bounds.split(' ')[0]),float(flux_bounds.split(' ')[1])),
          UniformPrior(float(va_bounds.split(' ')[0]),float(va_bounds.split(' ')[1])),
          UniformPrior(float(vm_bounds.split(' ')[0]),float(vm_bounds.split(' ')[1])))


# Define number of dimensions and number of walkers

ndim, nwalkers = 3, 100


# Define initial positions of walkers in phase space

frand = np.random.normal(loc=F_true,size=nwalkers,scale=0.1)
frand_2 = np.random.normal(loc=F_true,size=nwalkers,scale=0.1)
varand = np.random.normal(loc=va_true,size=nwalkers,scale=1.E3)
vmrand = np.random.normal(loc=vm_true,size=nwalkers,scale=1.E3)

pos = np.column_stack((frand,varand,vmrand)) 
pos_add_dim = np.expand_dims(pos,axis=0)
final_pos = np.repeat(pos_add_dim, 5, axis=0)

# Define initial positions for spectrum 4 walkers

pos_spec4 = np.column_stack((frand,frand_2,varand,vmrand)) 
pos_add_dim_spec4 = np.expand_dims(pos_spec4,axis=0)
final_pos_spec4 = np.repeat(pos_add_dim_spec4, 5, axis=0)

# Run MCMC sampler

sampler = emcee.PTSampler(5, nwalkers, ndim, lnlike, priors.lnprior, loglargs=[freqs,flux,error])
sams = sampler.run_mcmc(final_pos, 1000)


sampler_spec2 = emcee.PTSampler(5, nwalkers, ndim, lnlike_spec2, priors.lnprior, loglargs=[freqs,flux,error])
sams_spec2 = sampler_spec2.run_mcmc(final_pos, 1000)

sampler_spec3 = emcee.PTSampler(5, nwalkers, ndim, lnlike_spec3, priors.lnprior, loglargs=[freqs,flux,error])
sams_spec3 = sampler_spec3.run_mcmc(final_pos, 1000)

sampler_spec4 = emcee.PTSampler(5, nwalkers, 4, lnlike_spec4, priors_spec4.lnprior, loglargs=[freqs,flux,error])
sams_spec4 = sampler_spec4.run_mcmc(final_pos_spec4, 1000)


# Burn off initial steps
samples = sampler.chain[0,:, 500:, :].reshape((-1, ndim))

samples_spec2 = sampler_spec2.chain[0,:, 500:, :].reshape((-1, ndim))

samples_spec3 = sampler_spec3.chain[0,:, 500:, :].reshape((-1, ndim))

samples_spec4 = sampler_spec4.chain[0,:, 500:, :].reshape((-1, 4))


maxprobs = sampler.chain[0,...][np.where(sampler.lnprobability[0,...] == sampler.lnprobability[0,...].max())].mean(axis=0)

maxprobs_spec2 = sampler_spec2.chain[0,...][np.where(sampler_spec2.lnprobability[0,...] == sampler_spec2.lnprobability[0,...].max())].mean(axis=0)

maxprobs_spec3 = sampler_spec3.chain[0,...][np.where(sampler_spec3.lnprobability[0,...] == sampler_spec3.lnprobability[0,...].max())].mean(axis=0)

maxprobs_spec4 = sampler_spec4.chain[0,...][np.where(sampler_spec4.lnprobability[0,...] == sampler_spec4.lnprobability[0,...].max())].mean(axis=0)

# Plot corner plots if argument -c is passed

if plot_corner is True:
    fig = corner.corner(samples, labels=["$F_v$", "$v_a$", "$v_m$"],truths=[F_true,va_true,vm_true])


# Plot traces if argument -t is passed

if plot_traces is True:
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(sampler.chain[0, :, :, 0].T,color="k", alpha=0.3)
    plt.axhline(F_true, color='#4682b4')
    plt.ylabel('F_v')
    plt.xlabel('Number of Steps')
    
    plt.subplot(3,1,2)
    plt.plot(sampler.chain[0,:, :, 1].T, color="k", alpha=0.3)
    plt.axhline(va_true, color='#4682b4')
    plt.ylabel('v_a')
    plt.xlabel('Number of Steps')
    
    plt.subplot(3,1,3)
    plt.plot(sampler.chain[0,:, :, 2].T, color="k", alpha=0.3)
    plt.axhline(vm_true, color='#4682b4')
    plt.ylabel('v_m')
    plt.xlabel('Number of Steps')
    
    

# Print parameter estimation results corresponding to max of lnprob

F_mcmc, va_mcmc, vm_mcmc = maxprobs
F_spec2_mcmc, va_spec2_mcmc, vm_spec2_mcmc = maxprobs_spec2
F_spec3_mcmc, va_spec3_mcmc, vm_spec3_mcmc = maxprobs_spec3
F_spec4_mcmc, F2_spec4_mcmc,va_spec4_mcmc, vm_spec4_mcmc = maxprobs_spec4

# Print results
print "F_v_spec1 = %s" % F_mcmc
print "v_a_spec1 = %s" % va_mcmc
print "v_m_spec1 = %s" % vm_mcmc

print "F_v_spec2 = %s" % F_spec2_mcmc
print "v_a_spec2 = %s" % va_spec2_mcmc
print "v_m_spec2 = %s" % vm_spec2_mcmc

print "F_v_spec3 = %s" % F_spec3_mcmc
print "v_a_spec3 = %s" % va_spec3_mcmc
print "v_m_spec3 = %s" % vm_spec3_mcmc

print "F_v_spec4 = %s" % F_spec4_mcmc
print "F2_v_spec4 = %s" % F2_spec4_mcmc
print "v_a_spec4 = %s" % va_spec4_mcmc
print "v_m_spec4 = %s" % vm_spec4_mcmc

print "Log Likelihood = %s" %lnlike([F_mcmc,va_mcmc,vm_mcmc], freqs, flux, error)

v_range = np.linspace(1E9,350E9,1E4)
plt.figure()
plt.scatter(freqs,flux,color='k')
plt.plot(v_range,spectrum(v_range,F_mcmc,va_mcmc,vm_mcmc),lw='0.5',label='Spectrum 1')
plt.plot(v_range,spectrum_2(v_range,F_spec2_mcmc,va_spec2_mcmc,vm_spec2_mcmc),lw='0.8',label='Spectrum 2')
plt.plot(v_range,weighted_spectrum(v_range,F_spec3_mcmc,va_spec3_mcmc,vm_spec3_mcmc),ls='-.',lw='0.9',label='Weighted Spectrum (F1=F2)')
plt.plot(v_range,comb_spectrum(v_range,F_spec4_mcmc,F2_spec4_mcmc,va_spec4_mcmc,vm_spec4_mcmc),ls=':',lw='0.9',label='Weighted Spectrum')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.title(data_file.split('_')[1])
plt.show()
