#!/usr/bin/env python


import numpy as np

class UniformPrior:

    ''' Define a uniform (non-informative) prior for a single parameter.'''

    def __init__(self,lower_bound,upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    def lnprior(self, p):
        if (self.lower_bound < p < self.upper_bound):
            return 0.0
        return -np.inf

class FluxFrequencyPriors:

    ''' Assign priors for the flux normalization factor, self absorption frequency, and characteristic electron frequency.'''

    def __init__(self,flux_prior,va_prior,vm_prior):
        self.flux_prior = flux_prior
        self.va_prior = va_prior
        self.vm_prior = vm_prior

    def lnprior(self, flux, va, vm):
        return self.flux_prior.lnprior(flux) + self.va_prior.lnprior(va) + self.vm_prior.lnprior(vm)



