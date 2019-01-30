from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.stats import norm


def norm_percentiles(s):
    return 100 * norm.cdf(-s), 50, 100 * norm.cdf(s)


class zscore(BaseEstimator, TransformerMixin):
    """Transformer adapter that performs z-scoring similarly to `sklearn.preprocessing.StandardScaler`"""

    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler

    def __init__(self, mn=0.0, sd=1.0):
        self.mn = np.ravel(mn)
        self.sd = np.ravel(sd)

    def fit(self, X, *args, **kwargs):
        self.mn = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        return self

    def transform(self, X, *args, **kwargs):
        return (X - self.mn) / self.sd


class qscore(BaseEstimator, TransformerMixin):
    """Transformer adapter that performs z-scoring based on core data (according to quantiles that select outliers)
    similarly to `sklearn.preprocessing.RobustScaler`"""

    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler

    def __init__(self, mn=0.0, sd=1.0, outlier_level_sd=1.0):
        self.mn = np.ravel(mn)
        self.sd = np.ravel(sd)
        self.outlier_level_sd = float(outlier_level_sd)
        self._percentiles = norm_percentiles(outlier_level_sd)

    def fit(self, X, *args, **kwargs):
        q = np.nanpercentile(X, self._percentiles, axis=0)
        self.mn = q[1]
        self.sd = (q[2] - q[0]) / self.outlier_level_sd
        return self

    def transform(self, X, *args, **kwargs):
        return (X - self.mn) / self.sd


percentiles = {
    s: norm_percentiles(s) for s in range(4)
}
