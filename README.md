# SyncMCMC
Bayesian modeling of synchrotron emission using MCMC 

This package implements Parallel Tempered Markov Chain Monte Carlo methods in order to fit synchrotron spectra to emission from relativistic jets. The bulk Lorentz factor of the jet, number density of electrons, and physical source size are obtained from minimal energy equipartition arguments.

### Files

The following files are included in the repository.

- **model.ipynb**: Describes the the likelihood functions and models as implemented in the package.
- **tutorial.ipynb**: Demonstrates how to use the package and also specifies required data file formatting.
- **runmcmc.py**: Contains a function which executes the parallel-tempered MCMC sampler.
- **run_analysis.py**: An interactive script for analyzing a single time epoch of data. Allows the user to define various input parameters and perform diagnostics.
- **run_all_epochs.py**: Performs the full parameter estimate analysis on 16 epochs of data, produces plots, and outputs results to files.
- **plot_time_evolution.py**: Plots time evolution of parameter estimates from outputs produced by run_all_epochs.py. 
- **equipartition.py**: Calculates various physical parameters based on parameter estimates from outputs produced by run_all_epochs.py.
