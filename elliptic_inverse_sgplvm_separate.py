# Steven Atkinson
# satkinso@nd.edu
# June 1, 2018

"""
Build a surrogate for the elliptic problem using two SGPLVM models that are
trained separately.

Then, use it to solve the inverse problem

Model 1 is an unsupervised model of the inputs
Model 2 is a supervised model for the solution (single-output "pressure")
Model 2 uses the stochastic latent variables learned by Model 1 and keeps them
fixed during training.

See elliptic.util.parse_args() for options
"""

from __future__ import absolute_import

import elliptic

import os
import sys
from time import time
from warnings import warn
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    args = elliptic.util.parse_args()

    # Dimensionality of learned input space
    d_xi = min(args.n_train_in // 2, 128)
    # Inducing points
    m_in = min(args.n_train_in // 2, 128)
    m_out = min(args.n_train_out //2, 128)
    show_list = []  # i + 1 for i in range(10)]
    save_list = [i + 1 for i in range(10)]

    tf_config = tf.ConfigProto(
        intra_op_parallelism_threads=args.tf_num_threads,
        inter_op_parallelism_threads=args.tf_num_threads)
    sess = tf.Session(config=tf_config)
    with sess.as_default():
        tic = time()
        # Training data:
        # Output has a std of about 0.08 for problem 1 once the mean is removed.
        x_s, y_in_train = elliptic.data.load_inputs(args.train_dir,
            args.n_train_in, transform=args.transform_inputs)
        x_s, y_out_train = elliptic.data.load_outputs(args.train_dir,
            args.n_train_out)
        # Test data
        _, y_in_test = elliptic.data.load_inputs(args.test_dir, 
            args.n_test, transform=args.transform_inputs)
        x_s_infer, y_out_test = elliptic.data.load_outputs(
            args.test_dir, args.n_test, subsample=(0, args.obs_subsample), 
            noise=args.obs_noise)

        # Train models
        if args.xi_kern_in == "PCA":
            model_in = elliptic.train.build_pca(y_in_train, d_xi)
            mu, s = model_in.latent_variables()
            s += 1.0e-15
            mu = mu[:args.n_train_out]
            s = s[:args.n_train_out]
        else:
            model_in = elliptic.train.initialize_and_optimize_model(
                x_s, y_in_train, m_in, d_xi, 
                args.save_model_file_in, infer_type=args.infer_type, 
                xi_kern=args.xi_kern_in)
            mu = model_in.X0.value[:args.n_train_out]
            s = model_in.h_s.value[:args.n_train_out]
        model_out = elliptic.train.initialize_and_optimize_model(
            x_s, y_out_train, m_out, d_xi, 
            args.save_model_file_out, mu=mu, s=s, infer_type=args.infer_type, 
            xi_kern=args.xi_kern_out, x_st_infer=x_s_infer)

        elliptic.analysis.report_snr(model_in)
        elliptic.analysis.report_snr(model_out)

        # Predictions
        rmse, mnlp = elliptic.analysis.predict_backward(model_in, model_out,
            y_in_test, y_out_test, show=show_list, save_list=save_list,
            save_dir=args.save_dir, tf_config=tf_config)
    print("Prediction results:")
    elliptic.util.print_mean_and_std(rmse, "RMSE")
    elliptic.util.print_mean_and_std(mnlp, "MNLP")
    toc = time() - tic
    print("Total run time = {} sec".format(toc))
