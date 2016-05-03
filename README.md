# SyncMCMC
Bayesian modeling of synchrotron emission using MCMC 

This package implements Parallel Tempered Markov Chain Monte Carlo methods in order to fit synchrotron spectra to emission from relativistic jets. The bulk Lorentz factor of the jet, number density of electrons, and physical source size are obtained from minimal energy equipartition arguments.

### Files

The following files are included in the repository.

- **model.ipynb**: Describes the the likelihood functions and models as implemented in the package.
- **tutorial.ipynb**: Demonstrates how to use the package and also specifies required data file formatting.
- **runmcmc.py**: Contains a function which executes the parallel-tempered MCMC sampler.
- **run_analysis.py**: An interactive script for analyzing a single time epoch of data. Allows the user to define various input parameters and perform diagnostics.
- **run_all_epochs.py**: Performs the full parameter estimate analysis on 16 epochs of data, produces plots, and outputs results to files. Requires that paths to all data files are specified in the script.
- **plot_time_evolution.py**: Plots time evolution of parameter estimates from outputs produced by run_all_epochs.py. 
- **equipartition.py**: Calculates various physical parameters based on parameter estimates from outputs produced by run_all_epochs.py.

### Usage

It is highly recommended that the user runs the run_analysis.py script prior to exploring run_all_epochs.py. The former takes as input a single data file and allows for optional arguments for the priors and best guess values for the parameters. In executing run_all_epochs.py, all data files must be specified in the script, with the correct order corresponding to time from onset of the relativistic jet. 


### Data File Format

Data files should be formatted in a particular manner. Failure to properly format data files will prevent the analysis scripts from executing. All files should contain three columns listing the frequency, flux, and error in that order, and separated by commas. Any additional comment lines should include a '#'.
