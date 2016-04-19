
from unittest import TestCase
import unittest
import syncmcmc
from priorclasses import UniformPrior
from priorclasses import FluxFrequencyPriors


# Test UniformPrior Class

class UniformPriorTestCase(unittest.TestCase): 
   def test_UniformPriors(self):
      """Tests UniformPrior class"""
      # Define a uniform prior between 1 and 10
      test_uniform_priors = UniformPrior(1,10)
      # Is 5 contained within the prior bounds? If so, return 0
      self.assertFalse(test_uniform_priors.lnprior(5))
  
  
# Test FluxFrequencyPriors Class
        
class FluxFrequencyPriorsTestCase(unittest.TestCase):
   def test_FluxFrequencyPriors(self):
      '''Tests FluxFrequencyPriors class'''
      # Define uniform priors for flux, self absorption frequency, and characteristic frequency
      test_all_priors = FluxFrequencyPriors(UniformPrior(1.,50.),UniformPrior(1.E10,1.E13),UniformPrior(1.E10,1.E13))
      # Check if the following values are contained with prior bounds. If so, return 0
      self.assertFalse(test_all_priors.lnprior(10.,1.E11,1.E12))


if __name__ == '__main__':
    unittest.main()
