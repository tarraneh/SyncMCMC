import unittest
from unittest import TestCase
import syncmcmc
# Need to have ..priorclasses to tell it to look in parent directory
from ..priorclasses import UniformPrior
from ..priorclasses import FluxFrequencyPriors
from ..model import lnlike
import pandas as pd
import os
from ..read_data_io import load_data

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




if __name__ == '__main__':
    unittest.main()
