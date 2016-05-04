import unittest
from unittest import TestCase
import syncmcmc
from ..all_classes import UniformPrior
from ..all_classes import *
from ..model import *
import pandas as pd
import os
from ..read_data_io import load_data
import numpy as np
import emcee
from ..runmcmc import run_PTmcmc

# Test UniformPrior Class

class UniformPriorTestCase(TestCase):
   def test_UniformPriors(self):
      """Tests UniformPrior class"""
      # Define a uniform prior between 1 and 10
      test_uniform_priors = UniformPrior(1,10)
      # Is 5 contained within the prior bounds? If so, return 0
      self.assertFalse(test_uniform_priors.lnprior(5))

# Test FluxFrequencyPriors Class

class FluxFrequencyPriorsTestCase(TestCase):
   def test_FluxFrequencyPriors(self):
      '''Tests FluxFrequencyPriors class'''
      # Define uniform priors for flux, self absorption frequency, and characteristic frequency
      test_all_priors = FluxFrequencyPriors(UniformPrior(1.,50.),UniformPrior(1.E10,1.E13),UniformPrior(1.E10,1.E13),UniformPrior(-2,-0.01))
      # Check if the following values are contained with prior bounds. If so, return 0
      self.assertFalse(test_all_priors.lnprior((10.,1.E11,1.E12,-1)))



# Test BoundedGaussianPrior Class

class BoundedGaussianPriorTestCase(TestCase):
    def test_BoundedGaussianPrior(self):
      test = BoundedGaussianPrior(10,5,0,100)
      self.assertIsNotNone(test,test.lnprior(10))


# Test GaussianPrior

class GaussianPriorTestCase(TestCase):
    def test_GaussianPrior(self):
        test = GaussianPrior(10,5)
        self.assertIsNotNone(test,test.lnprior(9))


# Test that data files load 

def test_LoadData():
  filename = get_example_data_file_path('Sw1644+57_126days')
  data = load_data(filename) 
  assert data[1][0] == 1.4E9



def get_example_data_file_path(filename, data_dir='data'):
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    up_dir = os.path.split(start_dir)[0]
    data_dir = os.path.join(up_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)


# Test the inference framework

class ModelInferenceTestCase(TestCase):
  def TestModelInference(self):
    '''Tests inference framework using Model 1 spectrum'''
    ### Define known parameters ### 

    p = 2.5 
    epsilon_e = 0.1 * (p-2.)/(p-1.)
    epsilon_b = 0.1

    # Spectral slopes
    
    beta_1 = 2.
    beta_2 = 1./3.
    beta_3 = (1.-p)/2.
    beta5_1 = 5.0/2.0
    beta5_2 = (1.0 - p)/2.0

    # Shape of spectrum at each break
    
    s_1 = 1.5
    s_2 = (1.76 + 0.05*p)
    s_3 = (0.8 - 0.03*p)
    s_4 = 3.63 * p - 1.60
    s_5 = 1.25 - 0.18 * p

    filename = get_example_data_file_path('Sw1644+57_197days')
    flux, freqs, error = load_data(filename) 

    # Define priors

    priors = FluxFrequencyPriors(UniformPrior(1.,55.),
          UniformPrior(1E8,1E11),
          UniformPrior(1E8,1E11), 
          UniformPrior(-3,0.01))


    F_true = 10**1.56
    va_true = 10**9.83
    vm_true = 10**10.04
    
    # Define number of dimensions and number of walkers
    
    ndim, nwalkers = 4, 10
    
    # Define initial positions of walkers in phase space for models 1, 2, and 3
    
    frand = np.random.normal(loc=F_true,size=nwalkers,scale=0.1)
    varand = np.random.normal(loc=va_true,size=nwalkers,scale=1.E3)
    vmrand = np.random.normal(loc=vm_true,size=nwalkers,scale=1.E3)
    yerrand = np.random.normal(loc=-0.7,size=nwalkers,scale=0.1)
    
    pos = np.column_stack((frand,varand,vmrand,yerrand)) 
    pos_add_dim = np.expand_dims(pos,axis=0)
    final_pos = np.repeat(pos_add_dim, 2, axis=0)

    sampler, sams = run_PTmcmc(2,lnlike, priors, final_pos, ndim, nwalkers, logargs=[freqs,flux,error])
    samples = sampler.chain[0,:, 300:, :].reshape((-1, ndim))

    F_mcmc, va_mcmc, vm_mcmc,lnf_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

    # Check that inference produces reasonable estimate for F
    self.assertTrue(20. < F_mcmc[0] < 50.)


# Test that the models produce reasonable results

class TestModels(TestCase):
  def test_spectrum(self):
    F_v = 10.
    va = 1E10
    vm = 1E11
    v_range = np.linspace(1E9,350E9,1E4)
    model = spectrum(v_range,F_v,va,vm)
    shape = np.shape(model)
    self.assertTrue(shape == (10000,))

  def test_spectrum2(self):
    F_v = 10.
    va = 1E10
    vm = 1E11
    v_range = np.linspace(1E9,350E9,1E4)
    model = spectrum_2(v_range,F_v,va,vm)
    shape = np.shape(model)
    self.assertTrue(shape == (10000,))

  def test_weighted_spectrum(self):
    F_v = 10.
    va = 1E10
    vm = 1E11
    v_range = np.linspace(1E9,350E9,1E4)
    model = weighted_spectrum(v_range,F_v,va,vm)
    shape = np.shape(model)
    self.assertTrue(shape == (10000,))

  def test_spectrum2(self):
    F_v = 10.
    va = 1E10
    vm = 1E11
    v_range = np.linspace(1E9,350E9,1E4)
    model = comb_spectrum(v_range,F_v,F_v,va,vm)
    shape = np.shape(model)
    max_value = np.max(model)
    self.assertTrue(shape == (10000,))

# Test the likelihood functions

class TestLikelihoods(TestCase):
  def test_lnlike_model1(self):
    F_v = 10.
    va = 1E10
    vm = 1E11
    filename = get_example_data_file_path('Sw1644+57_126days')
    flux, freqs, error = load_data(filename) 
    s = lnlike([10,1E10,1E11,-0.7],flux,freqs,error)
    self.assertIsNotNone(s)

  def test_lnlike_model2(self):
    F_v = 10.
    va = 1E10
    vm = 1E11
    filename = get_example_data_file_path('Sw1644+57_126days')
    flux, freqs, error = load_data(filename) 
    s = lnlike_spec2([10,1E10,1E11,-0.7],flux,freqs,error)
    self.assertIsNotNone(s)

  def test_lnlike_model3(self):
    F_v = 10.
    va = 1E10
    vm = 1E11
    filename = get_example_data_file_path('Sw1644+57_126days')
    flux, freqs, error = load_data(filename) 
    s = lnlike_spec3([10,1E10,1E11,-0.7],flux,freqs,error)
    self.assertIsNotNone(s)


  def test_lnlike_model4(self):
    F_v = 10.
    va = 1E10
    vm = 1E11
    filename = get_example_data_file_path('Sw1644+57_126days')
    flux, freqs, error = load_data(filename) 
    s = lnlike_spec4([10,10,1E10,1E11,-0.7],flux,freqs,error)
    self.assertIsNotNone(s)

if __name__ == '__main__':
    unittest.main()
