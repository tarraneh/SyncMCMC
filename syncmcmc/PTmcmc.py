import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner

''' Use Parallel Tempering Ensemlbe to sample the posterior distribution'''

# Define number of dimensions and number of walkers

ndim, nwalkers = 4, 100


# Define initial positions of walkers in phase space

frand = np.random.normal(loc=F_true,size=nwalkers,scale=0.1)
varand = np.random.normal(loc=va_true,size=nwalkers,scale=1.E3)
vmrand = np.random.normal(loc=vm_true,size=nwalkers,scale=1.E3)
yerrand = np.random.normal(loc=-0.7,size=nwalkers,scale=0.1)

pos = np.column_stack((frand,varand,vmrand,yerrand)) 
pos1 = np.expand_dims(pos,axis=0)
pos2 = np.repeat(pos1, 5, axis=0)
sampler = emcee.PTSampler(5, nwalkers, ndim, lnlike, p.lnprior, loglargs=[v3,flux3,err3])
sams = sampler.run_mcmc(pos2, 1000)
