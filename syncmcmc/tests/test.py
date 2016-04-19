
from unittest import TestCase
import unittest
import syncmcmc
from data_class.py import UniformPrior

class TestFillerFunc(TestCase):
    def test_is_string(self):
        s = syncmcmc.fillerfunc()
        self.assertTrue(isinstance(s, str))

class UniformPriorTestCase(unittest.TestCase):
    """Tests UniformPrior class"""
    # Define a uniform prior between 1 and 10
    test = UniformPrior(1,10)
    # Is 5 contained within the prior bounds?
    self.assertTrue(test.lnprior(5))


if __name__ == '__main__':
    unittest.main()
