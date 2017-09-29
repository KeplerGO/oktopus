.. oktopus documentation master file, created by
   sphinx-quickstart on Tue Sep 26 09:14:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================
Welcome to 🐙 oktopus!
======================

**oktopus** is all about Bayes' Law:

.. math::

     \log \underbrace{p(\theta | \mathbf{y})}_\text{posterior} = \log \underbrace{p(\mathbf{y} | \theta)}_\text{likelihood} + \log \underbrace{p(\theta)}_\text{prior} + \overbrace{h(\mathbf{y})}^\text{doesn't depend on $\theta$}

In other words: **posterior** information is a combination of **prior** information and the information acquired after observing data (**likelihood**).

With that in mind, **oktopus** provides an easy interface to solve problems such as:

1. *Maximum Likelihood Estimator* (MLE):

.. math::

    \arg \min_{\theta \in \Theta} - \log p(\mathbf{y} | \theta)

2. *Fisher Information Matrix*:

.. math::

    \mathbb{E}\left[\nabla_\theta\log p(\mathbf{y} | \theta)\left[\nabla_\theta\log p(\mathbf{y} | \theta) \right]^{\textrm{T}}  \right]

3. *Maximum a Posteriori Probability Estimator* (MAP):

.. math::

    \arg \min_{\theta \in \Theta} - \log p(\theta | \mathbf{y})

*************
Documentation
*************

.. toctree::
    :maxdepth: 1

    install
    api/index
    ipython


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
