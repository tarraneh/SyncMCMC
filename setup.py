from setuptools import setup

setup(name='syncmcmc',
      version='0.1',
      description='Synchrotron Modeling using MCMC',
      url='http://github.com/phys201-sp2016/SyncMCMC',
      author='Tarraneh Eftekhari',
      author_email='teftekhari@cfa.harvard.edu',
      packages=['syncmcmc'],
      install_requires=[
          'numpy',
          'matplotlib',
          'emcee',
          'corner',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
