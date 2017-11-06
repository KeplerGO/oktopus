#!/usr/bin/env python
import sys
from setuptools import setup
# To use a consistent encoding
from codecs import open
import os
from os import path

# Load the version variable
exec(open('oktopus/version.py').read())

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/oktopus*")
    sys.exit()

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='oktopus',
    packages=['oktopus'],
    version=__version__,
    description='soft-bodied, eight-armed package for beautiful inference',
    long_description=long_description,
    url='https://github.com/mirca/oktopus',
    author='KeplerGO',
    author_email='jvmirca@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
    keywords='statistics probability',
    install_requires=['numpy', 'scipy', 'autograd']
)
