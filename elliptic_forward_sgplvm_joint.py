# Steven Atkinson
# satkinso@nd.edu
# May 23, 2018

"""
Build a surrogate for the elliptic problem using a joint SGPLVM model.
Then, use it to solve the forward problem.

Dimension 1 is for input data (conductivity)
Dimension 2 is for output data (the solution to the PDE)

BE SURE TO PROVIDE ARGUMENTS:
--joint_model 1
"""

import os
import sys
from time import time
from warnings import warn
import numpy as np
import tensorflow as tf

import elliptic


if __name__ == "__main__":
    args = elliptic.util.parse_args()
    assert args.joint_model, "Remember to specify joint=1"

    # Dimensionality of learned input space
    d_xi = min(args.n_train // 2, 128)
    # Inducing points
    m = min(args.n_train // 2, 128)
    show_list = []  # [i + 1 for i in range(10)]
    save_list = [i + 1 for i in range(10)]

    tf_config = tf.ConfigProto(
        intra_op_parallelism_threads=args.tf_num_threads,
        inter_op_parallelism_threads=args.tf_num_threads)
    sess = tf.Session(config=tf_config)
    with sess.as_default():
        tic = time()

        # Training data:
        x_s, y_in_train = elliptic.data.load_inputs(
            args.train_dir, args.n_train, transform=args.transform_inputs)
        _, y_out_train_raw = elliptic.data.load_outputs(
            args.train_dir, args.n_train)
        out_scale = np.std(y_out_train_raw)
        print("out_scale = {}".format(out_scale))
        y_out_train = y_out_train_raw / out_scale
        # Concatenate training data:
        y_train = np.concatenate((y_in_train, y_out_train), 1)
        # Test data
        _, y_in_test = elliptic.data.load_inputs(args.test_dir, args.n_test,
            transform=args.transform_inputs)
        # Rescaling for test data happens in the predict() routine
        x_s_infer, y_out_test = elliptic.data.load_outputs(args.test_dir, 
            args.n_test)

        # Train models
        model = elliptic.train.initialize_and_optimize_model(
            x_s, y_train, m, d_xi, args.save_model_file, 
            joint=True, infer_type=args.infer_type, xi_kern=args.xi_kern,
            x_st_infer=x_s_infer, opt_schedule=2)
        model_in, model_out = elliptic.analysis.split_model(
            model, xi_kern=args.xi_kern)

        # Predictions
        rmse, mnlp = elliptic.analysis.predict_forward(model_in, model_out,
                                                       y_in_test, y_out_test,
                                                       out_scale=out_scale,
                                                       show=show_list,
                                                       save_list=save_list,
                                                       save_dir=args.save_dir,
                                                       tf_config=tf_config, 
                                                       reconstruct=True)
    print("Prediction results:")
    elliptic.util.print_mean_and_std(rmse, "RMSE")
    elliptic.util.print_mean_and_std(mnlp, "MNLP")
    toc = time() - tic
    print("Total run time = {} sec".format(toc))
