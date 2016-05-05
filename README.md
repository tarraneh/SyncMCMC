# SyncMCMC

**Author**: Tarraneh Eftekhari (teftekhari@cfa.harvard.edu)

Bayesian modeling of synchrotron emission using MCMC 

This package implements Parallel Tempered Markov Chain Monte Carlo methods in order to fit synchrotron spectra to emission from relativistic jets. These parameters can be used via equipartition arguments to characterize physical properties of the jet, including the bulk Lorentz factor, total energy, and physical source size.

### Installation

The package can be downloaded and installed by running `python setup.py install` from the root directory. Tests can be run by passing the `nosetests` command from the root directory. These tests check that functions and inference methods within the package are working properly. 

### Files

The following files are included in the repository.

- `model.ipynb`: Describes the the likelihood functions and models as implemented in the package.
- `tutorial.ipynb`: Demonstrates how to use the package and also specifies required data file formatting.
- `runmcmc.py`: Contains a function which executes the parallel-tempered MCMC sampler.
- `run_analysis.py`: An interactive script for analyzing a single time epoch of data. Allows the user to define various input parameters and perform diagnostics.
- `run_all_epochs.py`: Performs the full parameter estimate analysis on 16 epochs of data, produces plots, and outputs results to files. Requires that paths to all data files are specified in the script.
- `plot_time_evolution.py`: Plots time evolution of parameter estimates from outputs produced by run_all_epochs.py. 
- `equipartition.py`: Calculates various physical parameters based on parameter estimates from outputs produced by run_all_epochs.py.
- `all_classes.py`: Contains classes used throughout the repository.
- `model.py`: Contains all four synchrotron models and their respective likelihood functions.
- `read_data_io.py`: Contains function which reads in and parses a data file.
- `runmcmc.py`: Contains function which executes parallel tempered Markov Chain Monte Carlo sampling


### Usage

It is highly recommended that the user runs the run_analysis.py script prior to exploring run_all_epochs.py. The former takes as input a single data file and allows for optional arguments for the priors and best guess values for the parameters. In executing run_all_epochs.py, all data files must be specified in the script, with the correct order corresponding to time since the onset of the relativistic jet. In addition, guess values for all parameters must be specified. For a detailed walkthrough of how to execute both scripts, see the `tutorial.ipynb` file in the root directory.


### Data File Format

Data files should be formatted in a particular manner. Failure to properly format data files will prevent the analysis scripts from executing. All files should contain three columns listing the frequency, flux, and error in that order, and separated by commas. Any additional comment lines should include a '#'. Please see `tutorial.ipynb` for details on the specific formatting.
