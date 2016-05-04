from all_classes import FluxFrequencyPriors, UniformPrior,FluxFrequencyPriorsCombinedSpectrum
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import corner
import argparse
import re
from model import *
from read_data_io import load_data
from runmcmc import run_PTmcmc
sns.set_style("white")




"""
   Run a Markov-Chain Monte Carlo sampler to determine best fit parameters for a synchrotron model. 


   Usage: run_analysis.py [options] -i filename

   Outputs best fit parameters for F_v, v_a, and v_m and a plot of the fit.

   Options:

   -i     --input        Specify data file
   -r     --raw          Plot raw data
   -fp    --fprior       Speicify lower and upper bounds for prior on flux normalization factor (Default: 1:50)
   -vap   --vaprior      Specify lower and upper bounds for prior on self absorption frequency (Default 1E9:1E11)
   -vam   --vamprior     Specify lower and upper bounds for prior on characteristic frequency (Default 1E9:1E11)
   -lnfp  --lnfprior     Specify lower and upper bounds for prior on fractional amount by which variance is underestimated
   -t     --trace        Plot MCMC traces for F_v, v_a, v_m
   -c     --corner       Plot corner plots
   -Ft     --F_true       Specify true value for F_v
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



# Read in and process command line options

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Specify input file for retrieving data', dest='data',type=str,default='None',action='store',required=True)
parser.add_argument('-r', '--raw', help='Plot raw data', dest='raw',default='None',action='store_true',required=False)
parser.add_argument('-fp', '--fprior', help='Define uniform prior bounds for flux normalization, -fp lower upper', dest='fluxprior',default='1.,110.',action='store',required=False)
parser.add_argument('-vap', '--vaprior', help='Define uniform prior bounds for self absorption frequency, -vap lower upper', dest='vaprior',type=str,default='1E9,1E12',action='store',required=False)
parser.add_argument('-vmp', '--vmprior', help='Define uniform prior bounds for characteristic frequency, -vam lower upper', dest='vmprior',type=str,default='1E9,1E12',action='store',required=False)
parser.add_argument('-lnfp' '--lnfprior', help='Define uniform prior bounds for fractional amount by which variance is underestimated, -lnfp lower upper', dest='lnfprior',type=str, default='-3.,-0.01',action='store',required=False)
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
lnf_bounds = args.lnfprior
plot_corner = args.corner
plot_traces = args.traces
F_true = (args.F_true)
va_true = (args.va_true)
vm_true = (args.vm_true)


# Load data

flux, freqs, error = load_data(data_file)


# Plot raw data if argument -r passed

if plot_raw_data is True:
   plt.figure()
   plt.scatter(freqs,flux)
   plt.xscale('log')
   plt.yscale('log')
   plt.show()


# Define priors

priors = FluxFrequencyPriors(UniformPrior(float(flux_bounds.split(',')[0]),float(flux_bounds.split(',')[1])),
          UniformPrior(float(va_bounds.split(',')[0]),float(va_bounds.split(',')[1])),
          UniformPrior(float(vm_bounds.split(',')[0]),float(vm_bounds.split(',')[1])), 
          UniformPrior(float(lnf_bounds.split(',')[0]),float(lnf_bounds.split(',')[1])))

priors_spec4 = FluxFrequencyPriorsCombinedSpectrum(UniformPrior(float(flux_bounds.split(',')[0]),float(flux_bounds.split(',')[1])),
          UniformPrior(float(flux_bounds.split(',')[0]),float(flux_bounds.split(',')[1])),
          UniformPrior(float(va_bounds.split(',')[0]),float(va_bounds.split(',')[1])),
          UniformPrior(float(vm_bounds.split(',')[0]),float(vm_bounds.split(',')[1])), 
          UniformPrior(float(lnf_bounds.split(',')[0]),float(lnf_bounds.split(',')[1])))



# Define number of dimensions and number of walkers

ndim, nwalkers = 4, 100



# Define initial positions of walkers in phase space for models 1, 2, and 3

frand = np.random.normal(loc=F_true,size=nwalkers,scale=0.1)
frand_2 = np.random.normal(loc=F_true,size=nwalkers,scale=0.1)
varand = np.random.normal(loc=va_true,size=nwalkers,scale=1.E3)
vmrand = np.random.normal(loc=vm_true,size=nwalkers,scale=1.E3)
yerrand = np.random.normal(loc=-0.7,size=nwalkers,scale=0.1)

pos = np.column_stack((frand,varand,vmrand,yerrand)) 
pos_add_dim = np.expand_dims(pos,axis=0)
final_pos = np.repeat(pos_add_dim, 5, axis=0)


# Define initial positions for spectrum 4 walkers

pos_spec4 = np.column_stack((frand,frand_2,varand,vmrand,yerrand)) 
pos_add_dim_spec4 = np.expand_dims(pos_spec4,axis=0)
final_pos_spec4 = np.repeat(pos_add_dim_spec4, 5, axis=0)


# Run MCMC sampler for each model

sampler, sams = run_PTmcmc(5,lnlike, priors, final_pos, ndim, nwalkers, logargs=[freqs,flux,error])

sampler_spec2, sams_spec2 = run_PTmcmc(5,lnlike_spec2, priors, final_pos, ndim, nwalkers, logargs=[freqs,flux,error])

sampler_spec3, sams_spec3 = run_PTmcmc(5,lnlike_spec3, priors, final_pos, ndim, nwalkers, logargs=[freqs,flux,error])

sampler_spec4, sams_spec4 = run_PTmcmc(5,lnlike_spec4, priors_spec4, final_pos_spec4, 5, nwalkers, logargs=[freqs,flux,error])



# Burn off initial steps
samples = sampler.chain[0,:, 500:, :].reshape((-1, ndim))

samples_spec2 = sampler_spec2.chain[0,:, 500:, :].reshape((-1, ndim))

samples_spec3 = sampler_spec3.chain[0,:, 500:, :].reshape((-1, ndim))

samples_spec4 = sampler_spec4.chain[0,:, 500:, :].reshape((-1, 5))


# Find values corresponding to maximum of posterior for each model

maxprobs = sampler.chain[0,...][np.where(sampler.lnprobability[0,...] == sampler.lnprobability[0,...].max())].mean(axis=0)

maxprobs_spec2 = sampler_spec2.chain[0,...][np.where(sampler_spec2.lnprobability[0,...] == sampler_spec2.lnprobability[0,...].max())].mean(axis=0)

maxprobs_spec3 = sampler_spec3.chain[0,...][np.where(sampler_spec3.lnprobability[0,...] == sampler_spec3.lnprobability[0,...].max())].mean(axis=0)

maxprobs_spec4 = sampler_spec4.chain[0,...][np.where(sampler_spec4.lnprobability[0,...] == sampler_spec4.lnprobability[0,...].max())].mean(axis=0)



# Plot corner plots if argument -c is passed

if plot_corner is True:
    fig = corner.corner(samples, labels=["$F_v$", "$v_a$", "$v_m$", "lnf"],truths=[F_true,va_true,vm_true, np.log(0.1)])


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

F_mcmc, va_mcmc, vm_mcmc, lnf_mcmc = maxprobs
F_spec2_mcmc, va_spec2_mcmc, vm_spec2_mcmc, lnf_spec2_mcmc = maxprobs_spec2
F_spec3_mcmc, va_spec3_mcmc, vm_spec3_mcmc, lnf_spec3_mcmc = maxprobs_spec3
F_spec4_mcmc, F2_spec4_mcmc,va_spec4_mcmc, vm_spec4_mcmc, lnf_spec4_mcmc = maxprobs_spec4


F_mcmc, va_mcmc, vm_mcmc,lnf_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))


F_spec2_mcmc, va_spec2_mcmc, vm_spec2_mcmc, lnf_spec2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_spec2, [16, 50, 84],axis=0)))

F_spec3_mcmc, va_spec3_mcmc, vm_spec3_mcmc,lnf_spec3_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_spec3, [16, 50, 84],axis=0)))

F_spec4_mcmc, F2_spec4_mcmc,va_spec4_mcmc, vm_spec4_mcmc,lnf_spec4_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_spec4, [16, 50, 84],axis=0)))

# Print results

print("""Model 1 MCMC result:
    F_v_spec1 = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    v_a_spec1 = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    v_m_spec1 = {4[0]} +{4[1]} -{4[2]} (truth: {5})
""".format(F_mcmc, F_true, va_mcmc, va_true, vm_mcmc, vm_true))

print("""Model 2 MCMC result:
    F_v_spec2 = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    v_a_spec2 = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    v_m_spec2 = {4[0]} +{4[1]} -{4[2]} (truth: {5})
""".format(F_spec2_mcmc, F_true, va_spec2_mcmc, va_true, vm_spec2_mcmc, vm_true))

print("""Model 3 MCMC result:
    F_v_spec3 = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    v_a_spec3 = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    v_m_spec3 = {4[0]} +{4[1]} -{4[2]} (truth: {5})
""".format(F_spec3_mcmc, F_true, va_spec3_mcmc, va_true, vm_spec3_mcmc, vm_true))

print("""Model 4 MCMC result:
    F_v_spec4 = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    v_a_spec4 = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    v_m_spec4 = {4[0]} +{4[1]} -{4[2]} (truth: {5})
""".format(F_spec4_mcmc, F_true, va_spec4_mcmc, va_true, vm_spec4_mcmc, vm_true))


# Write results to files
days = re.findall('\d+',data_file.split('_')[1])

spec1_results = [float(days[0]),F_mcmc[0],F_mcmc[1],F_mcmc[2],va_mcmc[0],va_mcmc[1],va_mcmc[2],vm_mcmc[0],vm_mcmc[1],vm_mcmc[2]]
spec2_results = [float(days[0]),F_spec2_mcmc[0],F_spec2_mcmc[1],F_spec2_mcmc[2],va_spec2_mcmc[0],va_spec2_mcmc[1],va_spec2_mcmc[2],
                vm_spec2_mcmc[0],vm_spec2_mcmc[1],vm_spec2_mcmc[2]]
spec3_results = [float(days[0]),F_spec3_mcmc[0],F_spec3_mcmc[1],F_spec3_mcmc[2],va_spec3_mcmc[0],va_spec3_mcmc[1],va_spec3_mcmc[2],
                vm_spec3_mcmc[0],vm_spec3_mcmc[1],vm_spec3_mcmc[2]]
spec4_results = [float(days[0]),F_spec4_mcmc[0],F_spec4_mcmc[1],F_spec4_mcmc[2],va_spec4_mcmc[0],va_spec4_mcmc[1],va_spec4_mcmc[2],
                vm_spec4_mcmc[0],vm_spec4_mcmc[1],vm_spec4_mcmc[2]]


with open("results/spectrum1_results_granotsari","a") as input_file:
    np.savetxt(input_file,spec1_results, fmt='%1.5f',newline=' ')
    input_file.write('\n')

with open("results/spectrum2_results_granotsari","a") as input_file:
    np.savetxt(input_file,spec2_results, fmt='%1.5f',newline=' ')
    input_file.write('\n')

with open("results/spectrum3_results_granotsari","a") as input_file:
    np.savetxt(input_file,spec3_results, fmt='%1.5f',newline=' ')
    input_file.write('\n')

with open("results/spectrum4_results_granotsari","a") as input_file:
    np.savetxt(input_file,spec4_results, fmt='%1.5f',newline=' ')
    input_file.write('\n')    


v_range = np.linspace(1E9,350E9,1E4)
plt.figure()
plt.scatter(freqs,flux,color='k')
plt.plot(v_range,spectrum(v_range,F_mcmc[0],va_mcmc[0],vm_mcmc[0]),color='#1b9e77',label='Spectrum 1',lw='0.9')
plt.plot(v_range,spectrum_2(v_range,F_spec2_mcmc[0],va_spec2_mcmc[0],vm_spec2_mcmc[0]),color='grey',label='Spectrum 2',lw='0.7')
plt.plot(v_range,weighted_spectrum(v_range,F_spec3_mcmc[0],va_spec3_mcmc[0],vm_spec3_mcmc[0]),ls='-.',color='#7570b3',label='Weighted Spectrum (F1=F2)')
plt.plot(v_range,comb_spectrum(v_range,F_spec4_mcmc[0],F2_spec4_mcmc[0],va_spec4_mcmc[0],vm_spec4_mcmc[0]),ls=':',color='#666666',label='Weighted Spectrum')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Flux [mJy]')
plt.title(data_file.split('_')[1])
plt.savefig('results/allfits_%s' %data_file.split('_')[1])
plt.show()

