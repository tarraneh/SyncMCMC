

from setuptools import setup

setup(name='syncmcmc',
      version='0.1',
      description='Fit synchrotron spectra to data using MCMC',
      url='https://github.com/p201-sp2016/SyncMCMC',
      author='Tarraneh Eftekhari',
      author_email='teftekhari@cfa.harvard.edu',
      packages=['syncmcmc'],
      install_requires=[
          'numpy'
          'matplotlib'
          'emcee'
      ],
      zip_safe=False)
