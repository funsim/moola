#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='moola',
      version='0.1.6',
      description='Moola optimisation package',
      author='Simon Funke',
      author_email='simon@simula.no',
      url='https://github.com/funsim/moola',
      packages=find_packages(),
      install_requires=[
          'numpy',
      ],
      classifiers=[
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Operating System :: OS Independent',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Mathematics'
      ]
)
