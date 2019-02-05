# Steven Atkinson
# satkinso@nd.edu
# April 9, 2018

"""
Data-driven SGPLVM elliptic predictions
"""

# from __future__ import absolute_import
import os
import inspect
model_path = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/../models"

from models import PCA
from .train import initialize_model

from structured_gpflow.models import Sgplvm, SgpModel, SgplvmFullCovInfer
from warnings import warn
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from time import time
import os


def _get_var(x):
    """
    Extract the variance of a matrix...

    If x is a 2D matrix with shape (nxm), then we take it to be the variances of
    n m-dimensional Gaussians.  Do nothing.

    If it's 3D with shape (nxmxm), then we have the mxm covariance matrices of n
    Gaussians.  Pull the diagonals out of them and reshape to be (nxm).

    :param x: [co]variances.
    :return: (nxm) variances.
    """
    return x if x.ndim == 2 else \
        np.array([np.diag(xi) for xi in x])


def _multivariate_normal(mu, lc):
    """
    Efficiently sample a multivariate normal

    :param mu: mean of dist (nxm)
    :param lc: (nxmxm), (lower) Cholesky of the covariance
    :return: (nxm) sample
    """
    return mu + (lc @ np.random.randn(*mu.shape, 1)).squeeze()


def _save_prediction(dir, y_in_i, y_out_i, y_mean, y_var, idx, mu_recon=None,
                     sigma_recon=None):
    """
    Saves matrices

    :param dir: Directory to be used
    :param y_in_i:
    :param y_out_i:
    :param y_mean:
    :param y_var:
    :param idx:
    :return:
    """
    if dir is None:
        dir = os.getcwd() + "/debug_predictions"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    filename_0 = "{}/test{}".format(dir, idx)

    if mu_recon is not None and sigma_recon is not None:
        recon_unc = 2.0 * np.sqrt(sigma_recon)
        recon_low = mu_recon - recon_unc
        recon_hi = mu_recon + recon_unc
    else:
        mu_recon, recon_low, recon_hi = None, None, None

    y_unc = 2.0 * np.sqrt(y_var)
    z_list = (y_in_i, recon_low, mu_recon, recon_hi,
              y_out_i, y_mean - y_unc, y_mean, y_mean + y_unc)
    name_list = ("in_true", "in_low", "in_mean", "in_hi",
                 "out_true", "out_low", "out_mean", "out_hi")
    for z, name in zip(z_list, name_list):
        if z is not None:
            np.savetxt("{}_{}.dat".format(filename_0, name), z)


def _sgplvm_forward_predict(model, h_sample_mtx=None, lv_mean=None, lv_cov=None,
                            n_samples=1, scale=1.0, use_test_noise=False):
    """
    Take
    :param model: Used to compute forward predictive densities
    :param h_sample_mtx: Provide set of samples explicitly
    :param lv_mean: (nxm) LV means
    :param lv_cov: Either
        1) nxm LV variances (diagonal Gaussians)
        2) nxmxm LV covarainces (full Gaussians)
    :param n_samples: How many samples to take.
    :param scale: rescale the predictions by this factor.
    :param use_test_noise: If true, use model.sigma2_infer to compute the
        forward prediction.
    :return: mean & variance of the predictive distribution.
    """

    # Initial predict: mainly to get the mean of the predictive distribution.
    # Operates on a Gaussian DISTRIBUTION 
    # when marginalizing over the input distribution.
    # Can be done analytically. 
    # Method to use depends on (use_test_noise, diagonal)
    initial_predict = {
        (False, True): model.predict_y,
        (False, False): model.predict_y_full_cov_in,
        (True, True): model.predict_y_sigma2_infer,
        (True, False): model.predict_y_sigma2_infer_full_cov_in
    }
    # Secondary predict function: used to get the covariance of the distribution
    # when marginalizing over the input distribution.
    # Operates on PARTICLES assumed to be samples from the input distribution.
    # Depends on (use_test_noise).
    second_predict = {
        True: model.predict_y_sigma2_infer,
        False: model.predict_y
    }

    samples_provided = h_sample_mtx is not None
    if samples_provided:
        n_samples = h_sample_mtx.shape[0]  # Overwrite anything provided
        assert lv_mean is None and lv_cov is None, \
            "Cannot provide samples as well as mean/var"
        if n_samples == 1:
            mu_y, sigma_y = second_predict[use_test_noise](
                h_sample_mtx[0], np.zeros(h_sample_mtx[0].shape))
    else:
        diagonal = lv_cov.ndim == 2
        if diagonal:
            lv_std = np.sqrt(lv_cov)
            h_sample_mtx = \
                lv_mean[np.newaxis, :, :] + lv_std[np.newaxis, :, :] * \
                np.random.randn(n_samples, *lv_mean.shape)
        else:
            lv_lcov = np.linalg.cholesky(lv_cov)
            h_sample_mtx = \
                np.array([
                    _multivariate_normal(lv_mean, lv_lcov)
                    for _ in range(n_samples)
                ])
        mu_y, sigma_y = initial_predict[(use_test_noise, diagonal)](lv_mean,
                                                                    lv_cov)
    if n_samples > 1:
        # Use points (zero variance):
        h_s_sample = np.zeros((1, model.d_xi))

        mu_sample_list, sigma_sample_list = [], []
        for h_mu_sample in h_sample_mtx:
            mu_sample, sigma_sample = second_predict[use_test_noise]\
                (h_mu_sample, h_s_sample)
            mu_sample_list.append(mu_sample)
            sigma_sample_list.append(sigma_sample)

        # If we didn't have MV, then we compute the mean numerically now:
        if samples_provided:
            mu_y = np.mean(np.array(mu_sample_list), 0)
        # Now compute sigma since we definitely have the mean.
        sigma_y = np.mean(np.array([
            (mu_sample - mu_y) ** 2 + sigma_sample
            for mu_sample, sigma_sample
            in zip(mu_sample_list, sigma_sample_list)]), 0)
    return mu_y * scale, sigma_y * scale ** 2


def _show_prediction(x_surf_in, y_surf_in, surf_shape_in, x_surf_out,
                     y_surf_out, surf_shape_out, y_in_i, mu_y, sigma_y, y_out_i,
                     n_in_train, n_out_train, idx):
    unc_y = 2.0 * np.sqrt(sigma_y)

    fig = plt.figure(figsize=(16, 4))
    for i, z_surf, title in zip(range(5),
                           (y_in_i, y_out_i, mu_y - unc_y, mu_y, mu_y + unc_y),
                           ("Test {}, input".format(idx), "True output",
                            "Lower confidence", "mean", "upper")):
        ax = fig.add_subplot(151 + i)
        if i == 0:
            x_surf = x_surf_in
            y_surf = y_surf_in
            surf_shape = surf_shape_in
        else:
            x_surf = x_surf_out
            y_surf = y_surf_out
            surf_shape = surf_shape_out
        im = ax.contourf(x_surf, y_surf, z_surf.reshape(surf_shape))
        fig.colorbar(im)
        # plt.title(title)
    plt.show()


def _surf(x, y, z, newfig=True, **kwargs):
    if newfig:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.contourf(x, y, z, **kwargs)
    fig.colorbar(im)


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def median_nlp(mu, sigma, x):
    """
    Element-wise median negative log probability

    :param mu: mean
    :param sigma: variance
    :param x: targets

    All inputs are np.ndarrays of the same size
    return: MNLP
    """
    p = 0.5 * (np.log(2.0 * np.pi) + np.log(sigma) + (x - mu)**2 / sigma)
    return np.median(p)


def get_surfs(model, use_infer=False):
    """
    Get the 2D matrices to be used for surf plots

    :param model:
    :return:
    """
    if isinstance(model, SgpModel):
        if use_infer:
            xs1, xs2 = getattr(model, model.input_st_infer_names[0]).value, \
                       getattr(model, model.input_st_infer_names[1]).value
        else:
            xs1, xs2 = model.X[1].value, model.X[2].value
        nsl = xs1.size
    else:
        ns = model.Y.shape[1]
        nsl = int(np.sqrt(ns))
        xs1 = np.linspace(0, 1, nsl).reshape(-1, 1)
        xs2 = xs1
    x_surf = np.tile(xs1, (1, nsl))
    y_surf = np.tile(xs2.T, (nsl, 1))
    surf_shape = (nsl, nsl)
    return x_surf, y_surf, surf_shape


def report_snr(model):
    """
    std(y) / sigma
    :param model:
    :return:
    """
    if isinstance(model, SgpModel):
        print("SNR = {}".format(np.std(model.Y.value) /
                                np.sqrt(model.likelihood.variance.value)))
    else:
        warn("No SNR implemented for model type {}".format(
            model.__class__.__name__))


def split_model(model, xi_kern=None):
    """
    Take a model where the output has multiple columns (channels) and split it
    so that each model only deals with a single column.
    :param model:
    :return: (list of models)
    """
    yt = model.Y.value.copy().T
    print("Split model ({} channels)...".format(yt.shape[0]))
    x_s = [model.X1.value.copy(), model.X2.value.copy()]
    # Spatials for inference of LV
    x_s_infer = [getattr(model, name).value
                 for name in model.input_st_infer_names]
    # Spatials for forward prediction
    x_s_test = [getattr(model, name).value
                for name in model.input_st_test_names]
    n_s = np.prod([x_si.shape[0] for x_si in x_s])
    infer_type = {
        Sgplvm: "diag",
        SgplvmFullCovInfer: "full"
    }[type(model)]
    models = []
    for i, y_col in enumerate(yt):
        y_col = y_col.reshape((-1, n_s))
        x_s_test_i = x_s_test if i == 0 else x_s_infer
        models.append(initialize_model(x_s, y_col, model.m_subgrid[0],
                                       model.num_latent, 
                                       infer_type=infer_type, xi_kern=xi_kern,
                                       x_st_infer=x_s_infer,
                                       x_st_test=x_s_test_i))
        models[-1].assign(model.read_trainables())
    return models


def predict_backward(model_in, model_out, y_in_test, y_out_test, **kwargs):
    return predict(model_out, model_in, y_out_test, y_in_test, False, **kwargs)


def predict_forward(model_in, model_out, y_in_test, y_out_test, **kwargs):
    return predict(model_in, model_out, y_in_test, y_out_test, True, **kwargs)


def predict(model_in, model_out, y_in_test, y_out_test, forward, out_scale=1.0,
            show=[], save_list=[], save_dir=None, n_samples=100,
            tf_config=None, reconstruct=False):
    """
    Infer a LV using model_in, then predict the output using model_out

    :param model_in:
    :param model_out:
    :param y_in_test:
    :param y_out_test:
    :param forward: (bool)
    :param show: Which ones to show.
    :param save_list: bool, for showing
    :param n_samples: for propagating latent variable uncertainty
    :param reconstruct: if True, try to reconstruct what you were provided!
    :return:
    """
    print("Predict test data...")

    n_in_train = model_in.X0.value.shape[0] if isinstance(model_in, SgpModel) \
        else model_in.n
    n_out_train = model_out.X0.value.shape[0] \
        if isinstance(model_out, SgpModel) else model_out.n
    n_test = len(y_in_test)

    x_surf_in, y_surf_in, surf_shape_in = get_surfs(model_in, use_infer=True)
    x_surf_out, y_surf_out, surf_shape_out = get_surfs(model_out)

    rmse_list = []
    median_nlp_list = []
    for i, y_in_i, y_out_i in zip(range(n_test), y_in_test, y_out_test):
        print("{} / {}".format(i + 1, n_test))
        save_this = i + 1 in save_list
        tic = time()
        graph = tf.Graph()
        if isinstance(model_in, SgpModel):
            model_in.clear()
        with graph.as_default():
            sess = tf.Session(graph=graph, config=tf_config)
            with sess.as_default():
                y_infer = y_in_i if forward else y_in_i / out_scale
                h_samples, h_mu_test, h_s_test = None, None, None
                if isinstance(model_in, Sgplvm):
                    h_mu_test, h_s_test = model_in.infer_pcb(
                        y_infer.reshape((1, -1)), test_noise=True)
                elif isinstance(model_in, PCA):
                    h_mu_test, h_s_test = model_in.infer(
                        y_infer.reshape((1, -1)))
                else:
                    raise ValueError("Input model {} not handled".format(
                        model_in.__class__.__name__))

                # Show the LV inference result:
                if h_mu_test is not None:
                    h_mu_test_show, h_s_test_show = h_mu_test, h_s_test
                print("LV:\n{}".
                      format(np.concatenate((h_mu_test_show.T,
                                             _get_var(h_s_test_show).T),
                                            1)))

                if reconstruct:
                    print("Reconstruct...")
                    recon_scale = 1.0 if forward else out_scale
                    mu_recon, sigma_recon = _sgplvm_forward_predict(
                        model_in, lv_mean=h_mu_test, lv_cov=h_s_test,
                        h_sample_mtx=h_samples, n_samples=n_samples,
                        scale=recon_scale, use_test_noise=True)
                else:
                    mu_recon, sigma_recon = None, None

        # Predict w/ sampling
        if np.max(h_s_test) == 0.0 or n_samples == 0 \
                or isinstance(model_out, PCA):
            print(" No sample needed")
            mu_y, sigma_y = model_out.predict_y(h_mu_test, _get_var(h_s_test))
        else:
            mu_y, sigma_y = _sgplvm_forward_predict(
                model_out, lv_mean=h_mu_test, lv_cov=h_s_test,
                h_sample_mtx=h_samples, n_samples=n_samples,
                use_test_noise=False)

        if forward:
            mu_y *= out_scale
            sigma_y *= out_scale

        toc = time() - tic
        print("Predict time = {}".format(toc))
        this_rmse = rmse(mu_y.flatten(), y_out_i.flatten())
        this_mnlp = median_nlp(mu_y.flatten(), sigma_y.flatten(),
                               y_out_i.flatten())
        print("RMSE={}\nMNLP={}".format(this_rmse, this_mnlp))
        rmse_list.append(this_rmse)
        median_nlp_list.append(this_mnlp)
        if i + 1 in show:
            _show_prediction(x_surf_in, y_surf_in, surf_shape_in, x_surf_out,
                             y_surf_out, surf_shape_out, y_in_i, mu_y, sigma_y,
                             y_out_i, n_in_train, n_out_train, i + 1)
        if save_this:
            _save_prediction(save_dir, y_in_i.reshape(surf_shape_in),
                             y_out_i.reshape(surf_shape_out),
                             mu_y.reshape(surf_shape_out),
                             sigma_y.reshape(surf_shape_out), i + 1,
                             mu_recon=mu_recon, sigma_recon=sigma_recon)
    return np.array(rmse_list), np.array(median_nlp_list)
