'''
Functions to visualize the results of benchmarks.
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import seaborn as sns

def heatmap_from_points(filename, X, y, vmin=None, vmax=None, colorbar=False):
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.clf()
        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
        grid_z0 = griddata(X, y, (grid_x, grid_y), method='nearest')
        plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
        plt.xlabel('$X_1$', fontsize=18, weight='bold')
        plt.ylabel('$X_2$', fontsize=18, weight='bold')
        if colorbar:
            plt.colorbar()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()



if __name__ == '__main__':
    from neural_sqerr import SqErrNetwork
    from neural_model import QuantileNetwork
    from spline_model import QuantileSpline
    from forest_model import QuantileForest
    results = np.load('data/mse_results.npy')[:25]
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    np.set_printoptions(precision=2, suppress=True, formatter={'float': lambda x: '{:.2f}'.format(x).ljust(10)} )

    averages = np.nanmean(results, axis=0)

    models = [SqErrNetwork(),
              QuantileNetwork(quantiles=quantiles),
              QuantileSpline(quantiles=quantiles),
              QuantileForest(quantiles=quantiles)]

    

    scenarios = [0,3,4,1,2]
    # for scenario in range(averages.shape[0]):
    for sidx, scenario in enumerate(scenarios):
        print(f'Scenario {sidx+1}')
        for nidx, n in enumerate([100, 1000, 10000]):
            print(f'n={n}')
            print('{:<20}{}'.format('Model', quantiles))
            for midx, model in enumerate(models):
                s = ' & '.join([f'{a:.2f}' for a in averages[scenario,midx,nidx]])
                print(f'& {model.label:<20} & {s} \\\\')
            print('')
        print('')
        print('')

    ##### Multivariate models
    results = np.load('data/multivariate_mse_results.npy')[:25]
    quantiles = np.array([0.5])
    averages = np.nanmean(results, axis=0)

    models = [SqErrNetwork(),
              QuantileNetwork(quantiles=quantiles),
              QuantileNetwork(quantiles=quantiles, loss='geometric'),
              # lambda: QuantileSpline(quantiles=quantiles),
              # lambda: QuantileForest(quantiles=quantiles)
              ]

    
    for scenario in range(averages.shape[0]):
        print(f'Scenario {scenario+1}')
        for nidx, n in enumerate([100, 1000, 10000]):
            print(f'n={n}')
            print('{:<35}{}'.format('Model', quantiles))
            for midx, model in enumerate(models):
                s = ' & '.join([f'{a:.2f}' for a in averages[scenario,midx,nidx]])
                print(f'& {model.label:<35} & {s} \\\\')
            print('')
        print('')
        print('')

















