#! /usr/bin/env python
from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages
from csdms.dakota import __version__
from csdms.dakota.run_plugin import plugin_script


setup(name='csdms-dakota',
      version=__version__,
      author='Mark Piper',
      author_email='mark.piper@colorado.edu',
      license='MIT',
      description='Python API for Dakota',
      long_description=open('README.md').read(),
      namespace_packages=['csdms'], 
      packages=find_packages(exclude=['*.tests']),
      entry_points={
          'console_scripts': [
              plugin_script + ' = csdms.dakota.run_plugin:main'
          ]
      }
)
