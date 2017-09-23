from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='oktopus',
    version='0.1.dev0',
    description='soft-bodied, eight-armed package for beautiful inference',
    long_description=long_description,
    url='https://github.com/mirca/octopus',
    author='Octopus developers',
    author_email='jvmirca@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Statistics/Inference',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='statistics probability',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy', 'scipy', 'autograd', 'astropy',
                      'pyketools']
)
