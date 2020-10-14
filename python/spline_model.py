import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrix, build_design_matrices
from pandas import DataFrame

class QuantileSpline:
    def __init__(self, quantiles=0.5, df=3):
        self.quantiles = quantiles
        self.df = df
        self.label = 'Quantile Spline'
        self.filename = 'spline'

    def fit(self, X, y):
        # Build the design matrix via a tensor basis expansion of natural spline bases
        data = {'x{}'.format(i+1): x for i, x in enumerate(X.T)}
        design_matrix = dmatrix("te(" +
                                    ",".join(['cr(x{}, df={})'.format(i+1, self.df) for i in range(X.shape[1])]) +
                                    ", constraints='center')", data)

        # Save the design information for future predictions
        self.design_info = design_matrix.design_info

        # Fit the model using the basis
        mod = smf.quantreg('y ~ x - 1', {'y': y, 'x': design_matrix})
        if np.isscalar(self.quantiles):
            self.model = mod.fit(q=self.quantiles)
        else:
            self.model = [mod.fit(q=q) for q in self.quantiles]

    def predict(self, X):
        data = {'x{}'.format(i+1): x for i, x in enumerate(X.T)}
        design_matrix = build_design_matrices([self.design_info], data)[0]
        if np.isscalar(self.quantiles):
            return self.model.predict({'x': design_matrix})
        return np.array([m.predict({'x': design_matrix}) for m in self.model]).T

    
