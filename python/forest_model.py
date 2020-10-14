import numpy as np
import six
import sys
sys.modules['sklearn.externals.six'] = six
from skgarden import RandomForestQuantileRegressor

class QuantileForest:
    def __init__(self, quantiles=0.5, min_samples_split=10, n_estimators=100):
        self.quantiles = quantiles
        self.model = RandomForestQuantileRegressor(random_state=0,
                                             min_samples_split=min_samples_split,
                                             n_estimators=n_estimators)
        self.label = 'Quantile Forest'
        self.filename = 'rf'


    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if np.isscalar(self.quantiles):
            return self.model.predict(X, quantile=self.quantiles*100)
        return np.array([self.model.predict(X, quantile=q*100) for q in self.quantiles]).T


