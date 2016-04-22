import numpy as np



from collections import OrderedDict
import inspect
import numpy as np

class Printable:
    @property
    def _dict(self):
        dump_dict = OrderedDict()

        for var in inspect.signature(self.__init__).parameters:
            if getattr(self, var, None) is not None:
                item = getattr(self, var)
                if isinstance(item, np.ndarray) and item.ndim == 1:
                    item = list(item)
                dump_dict[var] = item

        return dump_dict

    def __repr__(self):
        keywpairs = ["{0}={1}".format(k[0], repr(k[1])) for k in self._dict.items()]
        return "{0}({1})".format(self.__class__.__name__, ", ".join(keywpairs))

    def __str__(self):
        return self.__repr__()

class UniformPrior(Printable):

    ''' Define a uniform (non-informative) prior for a single parameter.'''

    def __init__(self,lower_bound,upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    def lnprior(self, p):
        if (self.lower_bound < p < self.upper_bound):
            return 0.0
        return -np.inf

class FluxFrequencyPriors(Printable):

    ''' Assign priors for the flux normalization factor, self absorption frequency, and characteristic electron frequency.'''

    def __init__(self,flux_prior,va_prior,vm_prior,lnf_prior):
        self.flux_prior = flux_prior
        self.va_prior = va_prior
        self.vm_prior = vm_prior
        self.lnf_prior = lnf_prior

    def lnprior(self, theta):
        flux, va, vm, lnf = theta
        return self.flux_prior.lnprior(flux) + self.va_prior.lnprior(va) + self.vm_prior.lnprior(vm) + self.lnf_prior.lnprior(lnf)


class FluxFrequencyPriorsCombinedSpectrum(Printable):

    ''' Assign priors for the flux normalization factor, self absorption frequency, and characteristic electron frequency.'''

    def __init__(self,flux_prior,flux2_prior,va_prior,vm_prior,lnf_prior):
        self.flux_prior = flux_prior
        self.flux2_prior = flux2_prior
        self.va_prior = va_prior
        self.vm_prior = vm_prior
        self.lnf_prior = lnf_prior

    def lnprior(self, theta):
        flux, flux2, va, vm, lnf = theta
        return self.flux_prior.lnprior(flux) + self.flux2_prior.lnprior(flux2) + self.va_prior.lnprior(va) + self.vm_prior.lnprior(vm) + self.lnf_prior.lnprior(lnf)

# Define priors for simple model (no correction for variance underestimates)

class FluxFrequencyPriorsSimpleModelCombinedSpectrum(Printable):

    ''' Assign priors for the flux normalization factor, self absorption frequency, and characteristic electron frequency.'''

    def __init__(self,flux_prior,flux2_prior,va_prior,vm_prior):
        self.flux_prior = flux_prior
        self.flux2_prior = flux2_prior
        self.va_prior = va_prior
        self.vm_prior = vm_prior


    def lnprior(self, theta):
        flux, flux2, va, vm = theta
        return self.flux_prior.lnprior(flux) + self.flux2_prior.lnprior(flux2) + self.va_prior.lnprior(va) + self.vm_prior.lnprior(vm)

class FluxFrequencyPriorsSimpleModel(Printable):

    ''' Assign priors for the flux normalization factor, self absorption frequency, and characteristic electron frequency.'''

    def __init__(self,flux_prior,va_prior,vm_prior):
        self.flux_prior = flux_prior
        self.va_prior = va_prior
        self.vm_prior = vm_prior


    def lnprior(self, theta):
        flux, va, vm = theta
        return self.flux_prior.lnprior(flux) + self.va_prior.lnprior(va) + self.vm_prior.lnprior(vm)
