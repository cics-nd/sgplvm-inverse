# Steven Atkinson
# satkinso@nd.edu
# May 23, 2018

"""
Simple PCA models
See Tipping & Bishop 1999
NOTE: default (rowvar=True) has data as columns, dimensions as rows (opposite of
"GP convention")

n = number of training data
d = data dimension
q = latent dimension
t = observed data
x = latent variables
Sample covariance s (normalization is by n)
"""

from __future__ import absolute_import

from structured_gpflow.util import np_least_squares
import numpy as np


class PCA(object):
    def __init__(self, t, q, rowvar=True, normalize=True):
        """

        :param t: Data
        :param q: Latent dimension
        :param rowvar: if True, then each datum is a column in t.
        :param normalize: if True, then the training data covariance will be an
            identity matrix.  Otherwise, the diagonals will be equal to the
            corresponding eigenvectors.
        """
        t_col = t if rowvar else t.T
        s = np.cov(t_col, bias=True)
        lam_asc, w_asc = np.linalg.eigh(s)  # Returned in ascending order
        lam, w = lam_asc[::-1][:q], w_asc[:, ::-1][:, :q]
        t_bar = np.mean(t_col, 1).reshape((-1, 1))

        self.t_bar = t_bar
        self.w = w
        self.scales = np.sqrt(lam).reshape((-1, 1))
        self.rowvar = rowvar
        self._normalize = normalize
        self._t = t
        self._x = self.infer(t)[0]

    @property
    def n(self):
        return self.x.shape[1] if self.rowvar else self.x.shape[0]

    @property
    def q(self):
        return self.w.shape[1]

    @property
    def x(self):
        return self._x

    @property
    def Y(self):
        return self._t

    @property
    def normalize(self):
        return self._normalize

    def infer(self, t, full_cov=False):
        """
        Project observations t to latent space
        :param t:
        :return: latent means, latent variances (zero!)
        """
        assert not full_cov, "No full_cov"
        t = t if self.rowvar else t.T
        x = self.w.T @ (t - self.t_bar)
        if self.normalize:
            x /= self.scales
        x = x if self.rowvar else x.T

        return x, np.zeros(x.shape)

    def latent_variables(self, full_cov=False):
        assert not full_cov, "No full_cov"
        return self.x, np.zeros(self.x.shape)

    def predict_y(self, mu, s_diag=None, full_cov=False):
        """
        Uplift to data space

        For rowvar=False, normalize=True:
        p(x) = N(x|mu, S)
        p(y|x) = N(y|A*x+b, C)
        p(y) = N(y|A*mu + b, C + A*S*A')

        Note that C=0 in this case because non-probabilistic PCA.

        :param mu:
        :param full_cov: whether to return full covariance matrix of predictive
            density
        :return: mean, var (or cov)
        """
        if s_diag is not None:
            assert any([s == 1 for s in s_diag.shape]), \
                    "One datum at a time for now"
        else:
            s_diag = np.zeros(mu.shape)
        if not self.rowvar:
            # Ensure both are data=columns:
            mu = mu.T
            s_diag = s_diag.T

        a = self.w if not self.normalize else self.w * self.scales.T
        b = self.t_bar
        # Mean of Gaussian
        t_mean = a @ mu + b
        # Variance of Gaussian:
        if full_cov:
            raise NotImplementedError("")
        else:
            # Actually variance
            # TODO need to do different broadcasting if we want to do more than
            # 1 at a time...
            t_cov = np.sum(s_diag.T * a ** 2, 1).reshape((-1, 1))

        if not self.rowvar:
            t_mean = t_mean.T
            t_cov = t_cov.T

        return t_mean, t_cov
