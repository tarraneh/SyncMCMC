
from unittest import TestCase

import syncmcmc

class TestFillerFunc(TestCase):
    def test_is_string(self):
        s = syncmcmc.fillerfunc()
        self.assertTrue(isinstance(s, str))
