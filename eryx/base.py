import numpy as np
import glob
import os

def reconstruct(ensemble_dir, n_grid_points, res_mask, n_asu):
    """
    Reconstruct a map from an ensemble of complex structure factors
    using Guinier's equation. The ensemble directory should contain
    a single file per asu / ensemble member, which are incoherently
    summed assuming that values at unmasked grid points were saved.
    
    Parameters
    ----------
    ensemble_dir : str
        path to directory containing ensemble's structure factors
    n_grid_points : int
        number of q-grid points, i.e. flattened map size
    res_mask : numpy.ndarray, shape (n_grid_points,)
        boolean resolution mask
    n_asu : int
        number of asymmetric units
    """
    fnames = glob.glob(os.path.join(ensemble_dir, "*npy"))
    n_ensemble = len(fnames) / n_asu
    
    fc = np.zeros(n_grid_points, dtype=complex)
    fc_square = np.zeros(n_grid_points)
    for fname in fnames:
        A = np.load(fname)
        fc[res_mask] += A
        fc_square[res_mask] += np.square(np.abs(A))
    Id = fc_square / n_ensemble - np.square(np.abs(fc / n_ensemble))
    
    Id[~res_mask] = np.nan
    return Id
