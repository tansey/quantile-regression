'''
Runs the quantile regression benchmarks for different models and functions.
'''
import numpy as np


from funcs import Sanity, Scenario1, Scenario2, Scenario3, Scenario4, Scenario5,\
                  MultivariateScenario1, MultivariateScenario2
from neural_sqerr import SqErrNetwork
from neural_model import QuantileNetwork
from spline_model import QuantileSpline
from forest_model import QuantileForest
from visualize import heatmap_from_points


def run_benchmarks(demo=True):
    N_trials = 100
    N_test = 10000
    sample_sizes = [100, 1000, 10000]
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    functions = [Scenario1(), Scenario2(), Scenario3(), Scenario4(), Scenario5()]
    models = [lambda: SqErrNetwork(),
              lambda: QuantileNetwork(quantiles=quantiles),
              lambda: QuantileSpline(quantiles=quantiles),
              lambda: QuantileForest(quantiles=quantiles)]

    # Track the performance results
    mse_results = np.full((N_trials, len(functions), len(models), len(sample_sizes), len(quantiles)), np.nan)
    print(mse_results.shape)

    for trial in range(N_trials):
        print(f'Trial {trial+1}')
        for scenario, func in enumerate(functions):
            print(f'\tScenario {scenario+1}')

            # Sample test set covariates and response
            X_test = np.random.random(size=(N_test,func.n_in))
            y_test = func.sample(X_test)

            # Get the ground truth quantiles
            y_quantiles = np.array([func.quantile(X_test, q) for q in quantiles]).T

            # Demo plotting
            if demo:
                for qidx, q in enumerate((quantiles*100).astype(int)):
                    heatmap_from_points(f'plots/scenario{scenario+1}-quantile{q}-truth.pdf', X_test[:,:2], y_quantiles[:,qidx], vmin=y_quantiles.min(), vmax=y_quantiles.max())

            for nidx, N_train in enumerate(sample_sizes):
                print(f'\t\tN={N_train}')
                # Sample training covariates and response
                X_train = np.random.random(size=(N_train,func.n_in))
                y_train = func.sample(X_train)

                # Evaluate each of the quantile models
                # Note: we generate a new model each time so as to not
                # accidentally cheat by warm-starting from the last point
                for midx, model in enumerate([m() for m in models]):
                    print(f'\t\t\t{model.label}')

                    if X_train.shape[1] > 3 and model.filename == 'spline':
                        print('Too many covariates. Skipping...')
                        continue

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # Evaluate the model on the ground truth quantiles
                    mse_results[trial, scenario, midx, nidx] = ((y_quantiles - preds)**2).mean(axis=0)

                    # Demo plotting
                    if demo:
                        # Plot the results for the first 2 coordinates
                        for qidx, q in enumerate((quantiles*100).astype(int)):
                            heatmap_from_points(f'plots/scenario{scenario+1}-quantile{q}-n{N_train}-{model.filename}.pdf', X_test[:,:2],
                                                    preds[:,qidx] if preds.shape[1] > qidx else preds[:,-1],
                                                    vmin=y_quantiles.min(), vmax=y_quantiles.max(),
                                                    colorbar=midx == len(models)-1)

            print('\t', mse_results[trial, scenario])

            if not demo:
                np.save('data/mse_results.npy', mse_results)

        if demo:
            return


def run_multivariate_benchmarks(demo=True):
    N_trials = 100
    N_test = 10000
    sample_sizes = [100, 1000, 10000]
    quantiles = np.array([0.5])
    functions = [MultivariateScenario1(), MultivariateScenario2()]
    models = [lambda: SqErrNetwork(),
              lambda: QuantileNetwork(quantiles=quantiles),
              lambda: QuantileNetwork(quantiles=quantiles, loss='geometric'),
              # lambda: QuantileSpline(quantiles=quantiles),
              # lambda: QuantileForest(quantiles=quantiles)
              ]

    # Track the performance results
    mse_results = np.full((N_trials, len(functions), len(models), len(sample_sizes), len(quantiles)), np.nan)
    print(mse_results.shape)

    for trial in range(N_trials):
        print(f'Trial {trial+1}')
        for scenario, func in enumerate(functions):
            print(f'\tScenario {scenario+1}')

            # Sample test set covariates and response
            X_test = np.random.random(size=(N_test,func.n_in))

            # Get the ground truth quantiles
            y_quantiles = np.transpose(np.array([func.quantile(X_test, q) for q in quantiles]), [1, 2, 0])
            print(y_quantiles.shape)

            for nidx, N_train in enumerate(sample_sizes):
                print(f'\t\tN={N_train}')
                # Sample training covariates and response
                X_train = np.random.random(size=(N_train,func.n_in))
                y_train = func.sample(X_train)

                # Evaluate each of the quantile models
                # Note: we generate a new model each time so as to not
                # accidentally cheat by warm-starting from the last point
                for midx, model in enumerate([m() for m in models]):
                    print(f'\t\t\t{model.label}')

                    if X_train.shape[1] > 3 and model.filename == 'spline':
                        print('Too many covariates. Skipping...')
                        continue

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # Evaluate the model on the ground truth quantiles
                    mse_results[trial, scenario, midx, nidx] = ((y_quantiles - preds)**2).mean(axis=0).mean(axis=0)

            print('\t', mse_results[trial, scenario])

            if not demo:
                np.save('data/multivariate_mse_results.npy', mse_results)
            else:
                pass



if __name__ == '__main__':
    # Reproducibility
    np.random.seed(42)
    import torch
    torch.manual_seed(42)

    np.set_printoptions(precision=2, suppress=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_benchmarks()
        # run_multivariate_benchmarks()













































