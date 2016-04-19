
from unittest import TestCase
import unittest
import syncmcmc
from priorclasses import UniformPrior


class UniformPriorTestCase(unittest.TestCase):
   def test_UniformPriors(self):
        """Tests UniformPrior class"""
        # Define a uniform prior between 1 and 10
        test = UniformPrior(1,10)
        # Is 5 contained within the prior bounds?
        self.assertFalse(test.lnprior(5))


if __name__ == '__main__':
    unittest.main()
