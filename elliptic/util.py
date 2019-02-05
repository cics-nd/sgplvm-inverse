# Steven Atkinson
# satkinso@nd.edu
# April 12, 2018

"""
Miscellaneous utilities
"""

from __future__ import absolute_import

import numpy as np
import pickle
import os
import argparse
from tempfile import mkdtemp


def get_local_config_param(key, force_update=False):
    """
    Read the local config file to find a parameter.  If it's not found, ask for
    it and store it for the future.

    :param key: What to get
    :param force_update: Ask the user regardless of if it's already been stored,
    and update what's in the file.
    :return:
    """
    val = None
    local_config_file = ".local.dat"
    local_config = {}
    if os.path.isfile(local_config_file):
        local_config.update(pickle.load(open(local_config_file, "rb")))
        val = local_config[key] if key in local_config else None
    if val is None or force_update:
        val = input("provide {}: ".format(key))
        local_config.update({key: val})
        pickle.dump(local_config, open(local_config_file, "wb"))
    return val


def print_mean_and_quantiles(x, name):
    """
    Just a quick pretty print

    :param x:
    :param name:
    :return:
    """
    print("{}: M{} Q({}, {})".format(name, np.mean(x),
        np.percentile(x, 2.5), np.percentile(x, 97.5)))


def print_mean_and_std(x, name):
    """
    Just a quick pretty print

    :param x:
    :param name:
    :return:
    """
    print("{}: {} ({})".format(name, np.mean(x), np.std(x)))


def parse_args():
    """
    Here's the argument parser that's used for all elliptic problem experiments

    :param joint_model: Whether we're using a joint-model approach or two-model.
        Two-model allows for different numbers of input and output training 
        examples.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_num_threads", type=int, default=1,
        help="Number of threads usable by TensorFlow")
    parser.add_argument("--joint_model", type=int, default=0,
        help="two-model (0) or joint model (1)")
    parser.add_argument("--infer_type", type=str, default="diag",
        choices=["diag", "full"],
        help="How to do LV inference")
    # Note: in/out/joint param defaults are defined below
    parser.add_argument("--xi_kern", type=str, default=None, 
        choices=["Linear", "RBF", "Sum"], 
        help="Stochastic kernel")
    parser.add_argument("--xi_kern_in", type=str, default=None,
        choices=["PCA", "Linear", "RBF", "Sum"], 
        help="Stochastic kernel (input submodel)")
    parser.add_argument("--xi_kern_out", type=str, default=None, 
        choices=["Linear", "RBF", "Sum"], 
        help="Stochastic kernel (output submodel)")
    parser.add_argument("--transform_inputs", type=int, default=1,
        help="Apply log transformation to inputs?")
    parser.add_argument("--n_train", type=int, default=None,
        help="Number of training examples")
    parser.add_argument("--n_train_in", type=int, default=None,
        help="Number of training input examples")
    parser.add_argument("--n_train_out", type=int, default=None,
        help="Number of training output examples")
    parser.add_argument("--save_model_file", type=str, default=None,
        help="Where to load/save the model")
    parser.add_argument("--save_model_file_in", type=str, default=None,
        help="Where to load/save the input model")
    parser.add_argument("--save_model_file_out", type=str, default=None,
        help="Where to load/save the output model")
    
    parser.add_argument("--n_test", type=int, default=100,
        help="Number of test examples to run")
    parser.add_argument("--train_dir", type=str, 
        default="elliptic/data/kl_16_32_train", 
        help="Directory where training input/ and output/ data are stored.")
    parser.add_argument("--test_dir", type=str,
        default="elliptic/data/kl_16_32_test", 
        help="Directory where test input/ and output/ data are stored.")
    parser.add_argument("--obs_subsample", type=int, default=1,
        help="Subsampling factor for test observations")
    parser.add_argument("--obs_noise", type=float, default=0.0,
        help="Std of white noise added to test observations")
    parser.add_argument("--save_dir", default=None, 
        help="Save directory for predictions")
    args = parser.parse_args()

    # Post-process and check arguments:
    
    # bools
    bool_args = ("joint_model", "transform_inputs")
    for arg in bool_args:
        setattr(args, arg, getattr(args, arg) == 1)

    # Parameters where 2M-vs-JM...
    def check_2m_jm_param(p_str, default_val):
        """
        1) Must define either both in/out or neither
        2) Must define AT MOST ONE of in/out or neither 
            (neither is OK--use default)
        3) Ensure in=out if joint model
        """  
        pin = getattr(args, p_str + "_in")
        pout = getattr(args, p_str + "_out")
        p = getattr(args, p_str)
        assert (pin is None) == (pout is None), \
            "Cannot specify one of n_train_in and n_train_out and not the " + \
            "other."
        assert int(pin is not None) + \
            int(p is not None) < 2, \
            "Cannot specify both n_train_in/out and n_train"
        if p is None:
            if pin is None:
                p = default_val
        if pin is None:
            pin, pout = p, p
        if args.joint_model:
            assert pin == pout, \
                "Must use same number of training inputs & outputs on " + \
                "joint model"  
        setattr(args, p_str + "_in", pin)
        setattr(args, p_str + "_out", pout)
        setattr(args, p_str, p)
    check_2m_jm_param("n_train", 16)
    check_2m_jm_param("xi_kern", "Linear")

    assert not (args.joint_model and args.xi_kern == "PCA"), \
        "Can't use PCA on joint model."
    
    # Where to save the model:
    if args.joint_model:
        assert args.save_model_file_in is None and \
            args.save_model_file_out is None, \
           "Joint model uses args.save_model_file"
    else:
        assert args.save_model_file is None, \
        "Two-model uses args.save_model_file_in and out"

    # Data: square np.ndarrays stored in .npy files
    def check_for_data(dir, n_in, n_out):
        for i in range(n_in):
            fname = os.path.join(dir, "inputs", "{}.npy".format(i))
            assert os.path.isfile(fname), "Didn't find input {}".format(fname)
        for i in range(n_out):
            fname = os.path.join(dir, "outputs", "{}.npy".format(i))
            assert os.path.isfile(fname), "Didn't find output {}".format(fname)
    check_for_data(args.train_dir, args.n_train_in, args.n_train_out)
    check_for_data(args.test_dir, args.n_test, args.n_test)

    # Get temporary save directory if needed:
    if args.save_dir is None:
        args.save_dir = mkdtemp()
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Print the args:
    print("Experiment args:")
    for key, val in args.__dict__.items():
        print(" {key:<24}: {val}".format(key=key, val=val))
    
    return args
