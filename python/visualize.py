'''
Functions to visualize the results of benchmarks.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

def heatmap_from_points(filename, X, y, vmin=None, vmax=None):
    plt.clf()
    print(X.shape, y.shape)
    # xi = np.linspace(0,1,20)
    # yi = np.linspace(0,1,20)
    # # grid the data.
    # zi = griddata(X[:,0], X[:,1], y, xi, yi, interp='linear')
    # print(zi)
    # plt.contourf(xi, yi, zi, vmax=y.max(), vmin=y.min(), cmap='gray_r', alpha=1, corner_mask=True)

    # grid_x1, grid_x2 = np.mgrid[0:1:200, 0:1:200]
    # grid_y = griddata(X, y, (grid_x1, grid_x2), method='nearest')
    # plt.imshow(grid_y.T, extent=(0,1,0,1), origin='lower', vmin=vmin, vmax=vmax)


    # m = plt.cm.ScalarMappable(cmap='gray_r')
    # m.set_array(zi)
    # m.set_clim(y.min(), y.max())
    # plt.colorbar(m, boundaries=np.linspace(y.min(),y.max(),31), alpha=1)
    # print(X.min(), X.max(), y.min(), y.max())

    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    grid_z0 = griddata(X, y, (grid_x, grid_y), method='nearest')
    plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


