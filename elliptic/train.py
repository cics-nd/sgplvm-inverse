# Steven Atkinson
# satkinso@nd.edu
# April 2, 2018

import sys
import os
import inspect
model_path = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/../models"

print("train: Add {} to path".format(model_path))
sys.path.append(model_path)
from models import PCA

from gpflow.kernels import RBF, Exponential, Linear
from structured_gpflow.models import SgpModel, Sgplvm, SgplvmFullCovInfer
from structured_gpflow.util import pca

import gpflow.kernels
from gpflow.priors import Gamma
from gpflow.training import ScipyOptimizer
from gpflow.training.tensorflow_optimizer import _TensorFlowOptimizer
from gpflow import transforms
from time import time
import numpy as np
from warnings import warn
import tempfile
import pickle

SKIP_TRAINING = False
MAX_TRIES = 10


def _perturb_lvs(model, supervised):
    """
    Shake up the latent variables a little

    :param model:
    :return:
    """
    def _perturb(_x):
        _x_val = _x.value
        scale = np.std(_x_val, 0)
        dx = np.random.randn(*_x_val.shape) * 0.05 * scale
        _x_val += dx
        _x =_x_val

    _perturb(model.feature.Z)
    if supervised:
        print("Note: no perturb of LVs for supervised models")
    else:
        _perturb(model.X0)


def _load_model(model, fname):
    print("Load model from {}".format(fname))
    params_loaded = pickle.load(open(fname, "rb"))

    # Check that the "model type" matches:
    model_type = model.pathname
    params_key0 = list(params_loaded.keys())[0]
    params_type = params_key0[:params_key0.find("/")]
    if model_type == params_type:
        params = params_loaded
    else:
        params = {}
        print("Replace model type {} with {}".format(params_type, model_type))
        for k_old in params_loaded.keys():
            k_new = k_old.replace(params_type, model_type)
            params[k_new] = params_loaded[k_old]
    # Assign
    # Allow incoherent params because the test-time inference params might not
    # match.
    model.assign(params, allow_incoherent=True)


def _lock_kerns(model):
    for kern in model.kern_list:
        kern.trainable = False


def _lock_lvs(model):
    model.X0.trainable = False
    model.h_s.trainable = False
    model.feature.trainable = False


def _unlock_kerns(model):
    model.kern_list[0].trainable = True
    for kern in model.kern_list[1:]:
        kern.lengthscales.trainable = True


def _unlock_lvs(model, supervised):
    if not supervised:
        model.X0.trainable = True
        model.h_s.trainable = True
    model.feature.trainable = True


def initialize_model(x_s, y, m, q, mu=None, s=None, joint=False, 
    infer_type="diag", xi_kern="RBF", x_st_infer=None, x_st_test=None):
    """

    :param use_kronecker:
    :param t: Inputs (i.e. simulator inputs)
    :param x_s: spatiotemporal inputs (just temporal for KO...)
    :type x_s: Iterable of 2D np.ndarrays
    :param y: outputs
    :param m:
    :param q:
    :param joint: if true then we're training a joint model with 2 channels
        (output dimensions).  Theya re assumed to be provided as
        column-concatenated in y.
    :param infer_type: How to infer latent variables.  Options:
        * "diag": VI with diagonal Gaussian variational posterior
        * "full": VI will full Gaussian variational posterior
        * "mcmc": MCMC particle inference
    :return:
    """

    n_s = np.prod([x_si.shape[0] for x_si in x_s])
    # Providing transformed variables...
    if mu is None:
        if joint:
            y_pca = y[:, :n_s]  # Inputs only
        else:
            y_pca = y
        mu = pca(y_pca, q)
        s = 0.1 * np.ones(mu.shape)
        supervised = False
        train_kl = True
    else:
        supervised = True
        train_kl = False
    if m == mu.shape[0]:
        z = mu
    else:
        z = None

    x = [mu] + x_s
    d_in = [x_i.shape[1] for x_i in x]

    with gpflow.defer_build():
        # X -> Y kernels
        if xi_kern == "Linear":
            kern_list = [Linear(d_in[0], variance=0.01, ARD=True)]
            kern_list[-1].variance.transform = transforms.Logistic(1.0e-12, 1.0)
        elif xi_kern == "RBF":
            kern_list = [RBF(d_in[0], lengthscales=np.sqrt(d_in[0]), ARD=True)]
            kern_list[-1].lengthscales.transform = transforms.Logistic(1.0e-12,
                                                                       1000.0)
            kern_list[-1].lengthscales.prior = Gamma(mu=2.0, var=1.0)
        elif xi_kern == "Sum":
            kern_list = [gpflow.kernels.Sum(
                [RBF(d_in[0], lengthscales=np.sqrt(d_in[0]), ARD=True),
                 Linear(d_in[0], variance=0.01, ARD=True)])]
            kern_list[-1].kernels[0].lengthscales.transform = \
                transforms.Logistic(1.0e-12, 1000.0)
            kern_list[-1].kernels[0].lengthscales.prior = Gamma(mu=2.0, var=1.0)
            kern_list[-1].kernels[1].variance.transform = \
                transforms.Logistic(1.0e-12, 1.0)
        else:
            raise NotImplementedError("Unknown xi kernel {}".format(xi_kern))
        for d_in_i in d_in[1:]:
            kern_list.append(Exponential(d_in_i, lengthscales=np.sqrt(d_in_i),
                                         ARD=True))
        for kern in kern_list[1:]:
            kern.lengthscales = 0.1

        # Restructure the inputs for the SGPLVM:
        if joint:
            y_structured = np.concatenate((y[:, :n_s].reshape((-1, 1)),
                                           y[:, n_s:].reshape((-1, 1))), 1)
        else:
            y_structured = y.reshape((-1, 1))

        # Initialize model:
        model_types = {
            "diag": Sgplvm,
            "full": SgplvmFullCovInfer
        }
        if infer_type not in model_types:
            raise NotImplementedError("No suport for infer_type {}".format(
                infer_type))
        else:
            kgplvm = model_types[infer_type]
        model = kgplvm(x, s, y_structured, kern_list, m, z, train_kl=train_kl,
                       x_st_infer=x_st_infer, x_st_test=x_st_test)
        model.likelihood.variance = 1.0e-2 * np.var(y)
        if supervised:
            # Lock provided inputs
            model.X0.trainable = False
            model.h_s.trainable = False
    model.compile()

    return model


def optimize_model(model, supervised, schedule=1, opt_iters=None):
    {1: optimize_model_1, 2: optimize_model_2}[schedule](model, supervised,
        opt_iters=opt_iters)


def optimize_model_1(model, supervised, opt_iters=None):
    """
    General optimization schedule for GPLVM models

    :param model:
    :return:
    """

    print("Optimize {} model...".format(
        model.__class__.__name__))
    t_start = time()
    model.compile()
    opt_success = False
    n_tries = 0
    method = "l-bfgs-b"
    # method = "cg"
    opt_iters = [100, 500, 1000] if opt_iters is None else opt_iters
    while not opt_success:
        try:
            n_tries += 1
            opt = ScipyOptimizer(method=method)

            # Session 1: lock all but LVs:
            model.likelihood.variance.trainable = False
            _lock_kerns(model)

            print("---Session 1: LVs only---")
            sess_idx = 1
            opt.minimize(model, disp=True, maxiter=opt_iters[0])
            sys.stdout.flush()
            print("After Session 1:")
            print(model)

            # Session 2: Unlock kerns:
            _unlock_kerns(model)
            model.compile()
            print("---Session 2: LVs, kerns---")
            sess_idx = 2
            opt.minimize(model, disp=True, maxiter=opt_iters[1])
            sys.stdout.flush()
            print("After Session 2:")
            print(model)

            # Session 2: Unlock noise
            model.likelihood.variance.trainable = True
            model.compile()
            print("---Session 3: All---")
            sess_idx = 3
            opt.minimize(model, disp=True, maxiter=opt_iters[2])
            sys.stdout.flush()
            print("After Session 3:")
            print(model)

            t_train = time() - t_start
            print("Training stage complete in {} seconds, {} tries.".format(
                t_train, n_tries))
            print("Trained model loss = {}".format(model.compute_objective()))
            opt_success = True
        except Exception as e:
            t_fail = time() - t_start
            print(
                "Optimization failed (attempt {}, session {}, t_fail={}): {}".
                    format(n_tries, sess_idx, t_fail, e))
            print("Current model state:")
            print(model)
            if n_tries == MAX_TRIES:
                raise RuntimeError("Max training tries exceeded")
            method = "cg"  # Switch to CG in case it works better?
            model.likelihood.variance = \
                2.0 * model.likelihood.variance.value
            print("Increase sigma2 to {}".format(
                model.likelihood.variance.value))
            print("Perturb...")
            _perturb_lvs(model, supervised)

    print("Training complete in {} tries".format(n_tries))


def optimize_model_2(model, supervised, opt_iters=None):
    """
    Kerns
    +LVs
    +Noise

    :param model:
    :param supervised: if true, then you can't move the LVs!
    :return:
    """

    print("Optimize {} model...".format(
        model.__class__.__name__))
    t_start = time()
    model.compile()
    opt_success = False
    n_tries = 0
    method = "l-bfgs-b"
    # method = "cg"
    opt_iters = [100, 500, 1000] if opt_iters is None else opt_iters
    while not opt_success:
        try:
            opt = ScipyOptimizer(method=method)
            n_tries += 1

            # Session 1: Kerns only (LVs locked):
            print("---Session 1: Kerns only---")
            sess_idx = 1
            model.likelihood.variance.trainable = False
            _lock_lvs(model)
            opt.minimize(model, disp=True, maxiter=opt_iters[0])
            sys.stdout.flush()
            print("After Session 1:")
            print(model)

            # Session 2: Unlock LVs:
            print("---Session 2: LVs/IPs, kerns---")
            sess_idx = 2
            _unlock_lvs(model, supervised)
            opt.minimize(model, disp=True, maxiter=opt_iters[1])
            sys.stdout.flush()
            print("After Session 2:")
            print(model)

            # Session 2: Unlock noise
            print("---Session 3: All---")
            sess_idx = 3
            model.likelihood.variance.trainable = True
            opt.minimize(model, disp=True, maxiter=opt_iters[2])
            sys.stdout.flush()
            print("After Session 3:")
            print(model)

            t_train = time() - t_start
            print("Training stage complete in {} seconds, {} tries.".format(
                t_train, n_tries))
            print("Trained model loss = {}".format(model.compute_objective()))
            opt_success = True
        except Exception as e:
            t_fail = time() - t_start
            print(
                "Optimization failed (attempt {}, session {}, t_fail={}): {}".
                  format(n_tries, sess_idx, t_fail, e))
            print("Current model state:")
            print(model)
            if n_tries == MAX_TRIES:
                raise RuntimeError("Max training tries exceeded")
            if method == "l-bfgs-b":
                method = "cg"  # Switch to CG in case it works better?
                opt_iters = [2 * x for x in opt_iters]  # Needs more iterations
            model.likelihood.variance = \
                2.0 * model.likelihood.variance.value
            print("Increase sigma2 to {}".format(
                model.likelihood.variance.value))
            print("Perturb...")
            _perturb_lvs(model, supervised)

    print("Training complete in {} tries".format(n_tries))


def print_kern_lengthscales(kern):
    if isinstance(kern, gpflow.kernels.Stationary):
        kern_ls = np.sort(kern.lengthscales.value)
    elif isinstance(kern, gpflow.kernels.Linear):
        kern_ls = np.sort(kern.variance.value)
    elif isinstance(kern, gpflow.kernels.Combination):
        for kern_i in kern.kernels:
            print_kern_lengthscales(kern_i)
        kern_ls = None
    else:
        kern_ls = None
    if kern_ls is not None:
        print("{}: {}".format(kern.name, kern_ls))


def save_model(model, fname):
    print("Save model to {}".format(fname))

    # Ensure the directory exists:
    if fname.rfind("/") != -1:
        fdir = fname[0: fname.rfind("/")]
        if not os.path.isdir(fdir):
            print("Create new directory {}".format(fdir))
            os.makedirs(fdir)

    params = model.read_trainables()
    pickle.dump(params, open(fname, "wb"))


def initialize_and_optimize_model(x_s, y, m, q, model_file, mu=None, s=None, 
    infer_type="diag", joint=False, xi_kern="RBF", x_st_infer=None,
    opt_schedule=2):
    """
    Get all of the initial guessing and such for a Sgplvm model, then train it.

    :param x_s: spatiotemporal inputs
    :param y: observed outputs [n_xi x n_s d_y]
    :param m:
    :param q: latent dimension
    :param mu: mean of latent inputs (supervised only)
    :param s: variance of latent inputs (supervised only)
    :param joint: whether we're joint-training on input & output data
        simultaneously.  If true, then we assume that they were provided in y as
        concatenated side-by-side.
    :param xi_kern: Which kernel to use for stochastics.  Choices:
        "RBF"
        "Linear"
        "Sum" (Linear + RBF)
    :param x_st_infer: points where we'll be doing inference
    :return:
    """
    model = initialize_model(x_s, y, m, q, mu=mu, s=s, joint=joint, 
        infer_type=infer_type, xi_kern=xi_kern, x_st_infer=x_st_infer)
    if model_file is not None and os.path.isfile(model_file):
        _load_model(model, model_file)
    else:
        if SKIP_TRAINING:
            warn("SKIP_TRAINING")
        else:
            supervised = mu is not None
            optimize_model(model, supervised, opt_schedule)
        # Save:
        if model_file is not None:
            save_model(model, model_file)

    print("---TRAINED MODEL---")
    print(model)
    print("(Sorted) Stochastic length scales:")
    if hasattr(model, "kern_list"):
        kern_0 = model.kern_list[0]
    else:
        kern_0 = model.kern
    print_kern_lengthscales(kern_0)
    sys.stdout.flush()

    return model


def build_pca(y, q):
    """
    Quick wrapper to build a PCA model instead of a GP-LVM
    :param y: data ("high dimension" n_xi x n_s)
    :param q: dimensions (int)
    :return:
    """
    return PCA(y, q, rowvar=False)
