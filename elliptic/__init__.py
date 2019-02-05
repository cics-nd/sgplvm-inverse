# Steven Atkinson
# satkinso@nd.edu
# April 5, 2018

"""
Elliptic problem
dimensionality reduction is data-driven and Bayesian, inputs to forward
surrogate are uncertain.

Uses two KGPLVMs
"""

from __future__ import absolute_import

from . import data, train, train_doe, analysis, util