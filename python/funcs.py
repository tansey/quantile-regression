'''
The collection of example simulated functions used in the paper.
'''
import numpy as np
from scipy.stats import t as t_dist, norm, cauchy, laplace

# # Sample some iid U(0,1) covariates
# X = np.random.random(size=(nsamples, 2))


class Benchmark:
    def __init__(self):
        pass

    def noiseless(self, X):
        raise NotImplementedError

    def quantile(self, X, q):
        raise NotImplementedError

    def sample(self, X):
        raise NotImplementedError

class Sanity(Benchmark):
    def __init__(self):
        super().__init__()

    def noiseless(self, X):
        return 10*X[:,0] + 5*X[:,1]

    def quantile(self, X, q):
        return 10*X[:,0] + 5*X[:,1] + norm.ppf(q)

    def sample(self, X):
        return 10*X[:,0] + 5*X[:,1] + norm.rvs(size=X.shape[0])    

class Scenario1(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 2

    def noiseless(self, X):
        return self.g2(self.g1(X))

    def quantile(self, X, q):
        return self.noiseless(X) + self.g3(X) * t_dist.ppf(q, 2)

    def sample(self, X):
        return self.noiseless(X) + self.g3(X) * np.random.standard_t(2, size=X.shape[0])

    def g1(self, X):
        return np.array([np.sqrt(X[:,0]) + X[:,0]*X[:,1], np.cos(2*np.pi*X[:,1])]).T

    def g2(self, X):
        return np.sqrt(X[:,0] + X[:,1]**2) + X[:,0]**2 * X[:,1]

    def g3(self, X):
        return np.linalg.norm(X - 0.5, axis=1)


class Scenario2(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 5

    def noiseless(self, X):
        return np.sqrt(X.sum(axis=1))

    def quantile(self, X, q):
        return self.noiseless(X) + laplace.ppf(q, scale=2)

    def sample(self, X):
        return self.noiseless(X) + laplace.rvs(scale=2, size=X.shape[0])

    

class Scenario3(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 10

    def noiseless(self, X):
        return self.g3(self.g2(self.g1(X)))

    def quantile(self, X, q):
        return self.noiseless(X) + t_dist.ppf(q, 3)

    def sample(self, X):
        return self.noiseless(X) + np.random.standard_t(3, size=X.shape[0])

    def g1(self, X):
        return np.array([np.sqrt(X[:,0]**2 + X[:,1:].sum(axis=1)),
                         X.sum(axis=1)**3]).T

    def g2(self, X):
        return np.array([np.abs(X[:,0]), np.prod(X, axis=1)]).T

    def g3(self, X):
        return X[:,0] + np.sqrt(X.sum(axis=1))


class Scenario4(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 2

    def noiseless(self, X):
        return (X**2).sum(axis=1)

    def quantile(self, X, q):
        return self.noiseless(X) + laplace.ppf(q, scale=2)

    def sample(self, X):
        return self.noiseless(X) + laplace.rvs(scale=2, size=X.shape[0])


class Scenario5(Benchmark):
    def __init__(self):
        super().__init__()
        self.beta = np.array([1, 0.5])
        self.n_in = 2

    def noiseless(self, X):
        return np.sqrt(X.sum(axis=1)) + (X[:,0] < 0.5).astype(float)

    def quantile(self, X, q):
        return self.noiseless(X) + np.sqrt(X.dot(self.beta)) * t_dist.ppf(q, 2)

    def sample(self, X):
        return self.noiseless(X) + np.sqrt(X.dot(self.beta)) * np.random.standard_t(2, size=X.shape[0])

class MultivariateScenario1(Benchmark):
    def __init__(self):
        super().__init__()
        self.beta = np.array([1, 0.5])
        self.n_in = 2

    def noiseless(self, X):
        return self.g2(self.g1(X))

    def quantile(self, X, q):
        return self.noiseless(X) + t_dist.ppf(q, 3)

    def sample(self, X):
        return self.noiseless(X) + np.random.standard_t(3, size=(X.shape[0], 2))

    def g1(self, X):
        return np.array([np.abs(X[:,0]), np.prod(X, axis=1)]).T

    def g2(self, X):
        return np.array([np.sqrt(X[:,0] + X[:,1]**2), X.sum(axis=1)**3]).T


class MultivariateScenario2(Benchmark):
    def __init__(self):
        super().__init__()
        self.beta = np.array([1, 0.5])
        self.n_in = 4

    def noiseless(self, X):
        X_sq = X**2
        return np.array([np.sqrt(X_sq[:,0] + X_sq[:,1]), np.sqrt(X_sq[:,2] + X_sq[:,3])]).T

    def quantile(self, X, q):
        return self.noiseless(X) + laplace.ppf(q, scale=2)

    def sample(self, X):
        return self.noiseless(X) + laplace.rvs(scale=2, size=(X.shape[0], 2))










