# Steven Atkinson
# satkinso@nd.edu
# March 19, 2018

"""
Utilities for loading data
"""

import os
import numpy as np

np.random.seed(42)
REMOVE_MEAN = True


def _load_data(dir, n, apply_log, remove_mean=False, subsample=None, noise=0.0):
    y_list = []
    x_s = None
    for i in range(n):
        filename = "{}/{}.npy".format(dir, i)
        this_file = np.load(filename)
        if i == 0:
            x_s = [np.linspace(0, 1, ns_i).reshape((-1, 1))
                   for ns_i in this_file.shape]
            if subsample is not None:
                s0, s1 = subsample
                x_s = [x_si[s0::s1, :] for x_si in x_s]
        if subsample is not None:
            this_file = this_file[s0::s1, s0::s1]
        if apply_log:
            this_file = np.log(this_file)
        if remove_mean:
            this_file -= 1.0 - x_s[0]
        y_list.append(this_file.flatten())
    y = np.array(y_list)
    y += noise * np.random.randn(*y.shape)
    return x_s, y


def load_inputs(dir, n, transform=True):
    return _load_data(os.path.join(dir, "inputs"), n, transform)


def load_outputs(dir, n, subsample=None, noise=0.0):
    return _load_data(os.path.join(dir, "outputs"), n, False, 
        remove_mean=REMOVE_MEAN, subsample=subsample, noise=noise)
