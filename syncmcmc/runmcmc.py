import emcee


"""

    Run a Parallel-Tempered Markov Chain Monte Carlo fitting algorithm.


    Options:

    ntemps          :: Number of temperatures
    lnlike          :: Log likelihood function
    priors          :: Array of priors for all parameters
    position        :: Array of initial position of walkers for all parameters
    ndim            :: Number of dimensions in parameter space
    nwalkers        :: Number of walkers your army contains
    logargs         :: Positional arguments for log likelihood function

"""



def run_PTmcmc(ntemps,lnlike, priors, position, ndim=3, nwalkers=100,logargs=[]):

    # Run MCMC sampler

    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, priors.lnprior, loglargs=logargs)
    sams = sampler.run_mcmc(position, 1000)

    return sampler, sams