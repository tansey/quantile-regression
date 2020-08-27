'''
The collection of example simulated functions used in the paper.
'''
import numpy as np
from scipy.stats import t as t_dist, norm

# # Sample some iid U(0,1) covariates
# X = np.random.random(size=(nsamples, 2))

def sanity_quantile(X, q):
    return 10*X[:,0] + 5*X[:,1] + norm.ppf(q)

def sanity_sample(X):
    return 10*X[:,0] + 5*X[:,1] + norm.rvs(size=X.shape[0])

def scenario1_quantile(X, q):
    def g1(q):
        return np.array([np.sqrt(q[:,0]) + q[:,0]*q[:,1], np.cos(2*np.pi*q[:,1])]).T

    def g2(q):
        return np.sqrt(q[:,0] + q[:,1]**2) + q[:,0]**2 * q[:,1]

    def g3(q):
        return np.linalg.norm(q - 0.5, axis=1)

    return g2(g1(X)) + g3(X) * t_dist.ppf(q, 2)

def scenario1_sample(X):
    def g1(q):
        return np.array([np.sqrt(q[:,0]) + q[:,0]*q[:,1], np.cos(2*np.pi*q[:,1])]).T

    def g2(q):
        return np.sqrt(q[:,0] + q[:,1]**2) + q[:,0]**2 * q[:,1]

    def g3(q):
        return np.linalg.norm(q - 0.5, axis=1)

    return g2(g1(X)) + g3(X) * np.random.standard_t(2, size=X.shape[0])













