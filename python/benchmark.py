'''
Runs the quantile regression benchmarks for different models and functions.
'''
import numpy as np


from funcs import scenario1_quantile, scenario1_sample, \
                  sanity_quantile, sanity_sample
from neural_model import QuantileNetworkModel, fit_quantiles
from spline_model import QuantileSpline
from forest_model import QuantileForest
from visualize import heatmap_from_points

if __name__ == '__main__':
    N = 20000
    test_pct = 0.5

    # Sample U(0,1)^2 covariates
    X = np.random.random(size=(N,2))

    # Sample responses
    # y = sanity_sample(X)
    # y_50 = sanity_quantile(X, 0.5)
    y = scenario1_sample(X)
    y_95 = scenario1_quantile(X, 0.95)
    y_75 = scenario1_quantile(X, 0.75)
    y_50 = scenario1_quantile(X, 0.5)
    y_25 = scenario1_quantile(X, 0.25)
    y_05 = scenario1_quantile(X, 0.05)

    # Split into train and test
    cutoff = int(N*test_pct)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    # Fit the models
    # nn_model = fit_quantiles(X_train, y_train, quantiles=quantiles, verbose=True)

    # spline_model = QuantileSpline(quantiles=quantiles)
    # spline_model.fit(X_train, y_train)

    forest_model = QuantileForest(quantiles=quantiles)
    forest_model.fit(X_train, y_train)

    # Get the held out predictions
    # nn_preds = nn_model.predict(X_test)
    # spline_preds = spline_model.predict(X_test)
    forest_preds = forest_model.predict(X_test)
    print(forest_preds.shape)


    # Plot the results
    heatmap_from_points('plots/scenario1_05.pdf', X_test, y_05[cutoff:], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_05_nn.pdf', X_test, nn_preds[:,0], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_05_spline.pdf', X_test, spline_preds[:,0], vmin=y_05.min(), vmax=y_95.max())
    heatmap_from_points('plots/scenario1_05_forest.pdf', X_test, forest_preds[:,0], vmin=y_05.min(), vmax=y_95.max())

    heatmap_from_points('plots/scenario1_25.pdf', X_test, y_25[cutoff:], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_25_nn.pdf', X_test, nn_preds[:,1], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_25_spline.pdf', X_test, spline_preds[:,1], vmin=y_05.min(), vmax=y_95.max())
    heatmap_from_points('plots/scenario1_25_forest.pdf', X_test, forest_preds[:,1], vmin=y_05.min(), vmax=y_95.max())

    heatmap_from_points('plots/scenario1_50.pdf', X_test, y_50[cutoff:], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_50_nn.pdf', X_test, nn_preds[:,2], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_50_spline.pdf', X_test, spline_preds[:,2], vmin=y_05.min(), vmax=y_95.max())
    heatmap_from_points('plots/scenario1_50_forest.pdf', X_test, forest_preds[:,2], vmin=y_05.min(), vmax=y_95.max())

    heatmap_from_points('plots/scenario1_75.pdf', X_test, y_75[cutoff:], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_75_nn.pdf', X_test, nn_preds[:,3], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_75_spline.pdf', X_test, spline_preds[:,3], vmin=y_05.min(), vmax=y_95.max())
    heatmap_from_points('plots/scenario1_75_forest.pdf', X_test, forest_preds[:,3], vmin=y_05.min(), vmax=y_95.max())

    heatmap_from_points('plots/scenario1_95.pdf', X_test, y_95[cutoff:], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_95_nn.pdf', X_test, nn_preds[:,4], vmin=y_05.min(), vmax=y_95.max())
    # heatmap_from_points('plots/scenario1_95_spline.pdf', X_test, spline_preds[:,4], vmin=y_05.min(), vmax=y_95.max())
    heatmap_from_points('plots/scenario1_95_forest.pdf', X_test, forest_preds[:,4], vmin=y_05.min(), vmax=y_95.max())


















































