from all_classes import FluxFrequencyPriors, UniformPrior,FluxFrequencyPriorsCombinedSpectrum, BoundedGaussianPrior, make_estimate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import corner
import argparse
import re
from model import *
from read_data_io import load_data
from runmcmc import run_PTmcmc
import time
sns.set_style("white")




"""
   Run a Markov-Chain Monte Carlo sampler to determine best fit parameters for a synchrotron model
   (three unique models are fit simultaneously -- see model.ipynb for details). This script runs the 
   analysis for multiple time epochs simultaneously, iteratively updating the priors for each subsequent 
   epoch. Data files must be specified below, and placed in the correct order corresponding to increasing 
   time since onset of the relativistic jet. Guess values should also be included for the flux, 
   self-absorption frequency, characteristic frequency, and uncertainty scaling factor for each epoch, 
   ie. number of guesses per array should equal the number of data files. 


   Usage: model.py [options]

   Outputs best fit parameters for F_v, v_a, and v_m and a plot of the fit.


   Options:
   -fp    --fprior       Speicify lower and upper bounds for prior on flux normalization factor (Default: 1:50)
   -vap   --vaprior      Specify lower and upper bounds for prior on self absorption frequency (Default 1E9:1E12)
   -vam   --vamprior     Specify lower and upper bounds for prior on characteristic frequency (Default 1E9:1E12)

   
   Fit produces F_v, v_a, v_m, lnf
   
   F_v               ::  Flux normalization 
   v_a               ::  Self-absorption frequency
   v_m               ::  Peak frequency
   lnf               ::  Fractional amount which uncertainties are underestimated by

"""



# Read in and process command line options

parser = argparse.ArgumentParser()
parser.add_argument('-fp', '--fprior', help='Define uniform prior bounds for flux normalization, -fp lower upper', dest='fluxprior',default='1.,110.',action='store',required=False)
parser.add_argument('-vap', '--vaprior', help='Define uniform prior bounds for self absorption frequency, -vap lower upper', dest='vaprior',type=str,default='1E9,1E12',action='store',required=False)
parser.add_argument('-vmp', '--vmprior', help='Define uniform prior bounds for characteristic frequency, -vam lower upper', dest='vmprior',type=str,default='1E9,1E12',action='store',required=False)


args = parser.parse_args()
flux_bounds = args.fluxprior
va_bounds = args.vaprior
vm_bounds = args.vmprior




### Load data ###

data_files = ['data/Sw1644+57_5days','data/Sw1644+57_10days','data/Sw1644+57_15days','data/Sw1644+57_22days',
              'data/Sw1644+57_36days','data/Sw1644+57_51days','data/Sw1644+57_68days','data/Sw1644+57_97days',
              'data/Sw1644+57_126days','data/Sw1644+57_161days','data/Sw1644+57_197days','data/Sw1644+57_216days',
              'data/Sw1644+57_244days','data/Sw1644+57_390days','data/Sw1644+57_457days','data/Sw1644+57_582days']



# Define guess values for initial iteration, all models

true_flux_values = [30.,9.,10.,9.,13.,16.,25.,33.,38.,33.,36.,43.,98.,52.,44.,40.]
true_va_values = [1E11,1E10,1E10,1E10,1E10,1E10,1E10,1E10,1E10,1E10,1E10,1E10,1E10,1E10,1E10,1E10]
true_vm_values = [1E12,1E11,1E11,1E11,1E11,1E11,1E10,1E10,1E10,1E10,1E10,1E10,1E10,1E9,1E9,1E9]
true_lnf_values = [-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7]



### Define priors ###
# va, vm, and lnf take on uniform priors
# F takes on a Gaussian prior with mean at the max data value

priors_1 = [UniformPrior(float(flux_bounds.split(',')[0]),float(flux_bounds.split(',')[1])),
          UniformPrior(float(va_bounds.split(',')[0]),float(va_bounds.split(',')[1])),
          UniformPrior(float(vm_bounds.split(',')[0]),float(vm_bounds.split(',')[1])), 
          UniformPrior(-3.,-0.01)]


# Create a list of priors, one for each model

priors_list = [priors_1, priors_1, priors_1]



# Create array of likelihood functions corresponding to each model

lnlikes = [lnlike, lnlike_spec2, lnlike_spec3]


# Begin parameter estimation for each data file and model

for (data_file, F_true, va_true, vm_true,lnf_true) in zip(data_files, true_flux_values, true_va_values, true_vm_values,true_lnf_values):
  start_epoch = time.time() # Start timing 

  flux,freqs, error = load_data(data_file) # Parse data


  # Combine priors for all parameters into a single array

  priors = [FluxFrequencyPriors(*p) for p in priors_list]


  # Define number of dimensions, number of walkers, and number of temperatures 
  
  ndim, nwalkers, ntemps = 4, 100, 5


  # Define initial positions of walkers in phase space for models 1, 2, and 3
  
  frand = np.random.normal(loc=F_true,size=nwalkers,scale=0.1)
  varand = np.random.normal(loc=va_true,size=nwalkers,scale=1.E3)
  vmrand = np.random.normal(loc=vm_true,size=nwalkers,scale=1.E3)
  yerrand = np.random.normal(loc=-0.7,size=nwalkers,scale=0.1)
  

  pos = np.column_stack((frand,varand,vmrand,yerrand)) 
  pos_add_dim = np.expand_dims(pos,axis=0)
  final_pos = np.repeat(pos_add_dim, 5, axis=0)


  all_samples = []
  #flux_stdevs = []
  
  all_model_ests = []

  # Run the MCMC sampler for all three models for a given time epoch

  for (prior, lnlike, mod_name) in zip(priors, lnlikes, range(1,4)):
    
    sampler, sams = run_PTmcmc(ntemps, lnlike, prior, final_pos, ndim, nwalkers,logargs=[freqs,flux,error])


    # Burn off initial steps
    samples = sampler.chain[0,:, 500:, :].reshape((-1, ndim))
    all_samples.append(samples)


    # Calculate parameter estimates and uncertainties
    estimates = [make_estimate(s, g) for (s, g) in zip(samples.T, (F_true,va_true,vm_true,lnf_true))]
    all_file_line = [make_estimate(s, g).get_value() for (s, g) in zip(samples.T, (F_true,va_true,vm_true,lnf_true))]
    all_model_ests.append(all_file_line)
    flux_est, va_est, vm_est, lnf_est = estimates


    print("""Model {name}:
      F = {e[0]}
      v_a = {e[1]}
      v_m = {e[2]}
      lnf = {e[3]}
      """.format(e=estimates, name=mod_name))
  
  
  
    # Write results to files

    days = re.findall('\d+',data_file.split('_')[1])
  
    with open("results/spectrum{}_results_phys201".format(mod_name),"a") as input_file:
      input_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(days[0], all_model_ests[0][0][0], all_model_ests[0][0][1], all_model_ests[0][0][2],
                                                                         all_model_ests[0][1][0], all_model_ests[0][1][1], all_model_ests[0][1][2],
                                                                         all_model_ests[0][2][0], all_model_ests[0][2][1], all_model_ests[0][2][2]))


    # Update priors with results from previous parameter estimation 

    #flux_stdevs.append(estimates[0].estimate_uncertainty())
    priors_list.append([UniformPrior(float(flux_bounds.split(',')[0]),float(flux_bounds.split(',')[1])),
          UniformPrior(float(va_bounds.split(',')[0]), va_est.upper),
          UniformPrior(float(vm_bounds.split(',')[0]), vm_est.upper),
          UniformPrior(-3.,-0.01)])
    
    end_epoch = time.time()
    print("Epoch took {}".format(end_epoch-start_epoch))



  # Plot fits from all models on a single figure 

  v_range = np.linspace(1E9,350E9,1E4)
  plt.figure()
  plt.scatter(freqs,flux,color='k')
  plt.plot(v_range,spectrum(v_range,all_model_ests[0][0][0],all_model_ests[0][1][0],all_model_ests[0][2][0]),color='#d95f02',label='Spectrum 1',lw='0.9')
  plt.plot(v_range,spectrum_2(v_range,all_model_ests[1][0][0],all_model_ests[1][1][0],all_model_ests[1][2][0]),ls='-.',color='#1b9e77',label='Spectrum 2')
  plt.plot(v_range,weighted_spectrum(v_range,all_model_ests[2][0][0],all_model_ests[2][1][0],all_model_ests[2][2][0]),ls=':',color='k',label='Weighted Spectrum')
  plt.legend(loc='lower right')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('Flux [mJy]')
  plt.title(data_file.split('_')[1])
  plt.savefig('results/allfits_test%s' %data_file.split('_')[1])
  