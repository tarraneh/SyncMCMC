import numpy as np
from scipy import stats

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


# Define uniform priors

class UniformPrior(Printable):

    ''' Define a uniform (non-informative) prior for a single parameter.'''

    def __init__(self,lower_bound,upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    def lnprior(self, p):
        if (self.lower_bound < p < self.upper_bound):
            return 0.0
        return -np.inf


# Return a list of all priors 

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


# Return a list of all priors for Model 4

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


# Define a Gaussian Prior

class GaussianPrior(Printable):
    def __init__(self, mu, sd):
        self.mu = mu
        self.sd = sd

    def lnprior(self, p):
        return stats.norm.logpdf(p, self.mu, self.sd)

    def sample(self, size=None):
        return random.normal(self.mu, self.sd, size=size)


#Bound a Gaussian prior

class BoundedGaussianPrior(GaussianPrior):
    # Note: this is not normalized
    def __init__(self, mu, sd, lower_bound=-np.inf, upper_bound=np.inf):
        if mu < lower_bound or mu > upper_bound:
            raise OutOfBoundsError(mu, lower_bound, upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mu = mu
        self.sd = sd

    def lnprior(self, p):
        p = p
        if p < self.lower_bound or p > self.upper_bound:
            return -np.inf
        else:
            return stats.norm.logpdf(p, self.mu, self.sd)


    def sample(self, size=None):
        val = super(BoundedGaussianPrior, self).sample(size)
        # TODO: do something smarter than just clipping
        return np.clip(val, self.lower_bound, self.upper_bound)

        return val


# Calculate parameter estimates and uncertainties

def make_estimate(samples, guess):
    lo, med, hi = np.percentile(samples, [16, 50, 84], axis=0)
    return ParameterEstimate(med, med-lo, hi-med, guess)


# Return string of estimates and uncertainties

class ParameterEstimate(Printable):
    def __init__(self,value,minus_uncertainty,plus_uncertainty,guess):
        self.value = value
        self.plus_uncertainty = plus_uncertainty
        self.minus_uncertainty = minus_uncertainty
        self.guess = guess

    def __str__(self):
        return "{s.value} +{s.plus_uncertainty} -{s.minus_uncertainty} (Guess: {s.guess})".format(s=self)

    # Return a line of estimates and uncertainties appropriate for copying to file

    def file_line(self):
        return "{s.value} {s.plus_uncertainty} {s.minus_uncertainty}".format(s=self)

    def get_value(self):
        return self.value, self.plus_uncertainty, self.minus_uncertainty

    # Grab the uncertainty bound which is greater
    def estimate_uncertainty(self):
        return np.max((self.plus_uncertainty,self.minus_uncertainty))

    @property
    def upper(self):
        return self.value + self.plus_uncertainty
    @property
    def lower(self):
        return self.value - self.minus_uncertainty    


