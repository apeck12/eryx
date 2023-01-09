import numpy as np

def pearson_cc(arr1, arr2):
    """
    Compute the Pearson correlation-coefficient between the input arrays.
    Voxels that should be ignored are assumed to have a value of NaN.
    
    Parameters
    ----------
    arr1 : numpy.ndarray, shape (n_samples, n_points)
        input array
    arr2 : numpy.ndarray, shape (n_samples, n_points) or (1, n_points)
        input array to compute CC with
    
    Returns
    -------
    ccs : numpy.ndarray, shape (n_samples)
        correlation coefficient between paired sample arrays, or if
        arr2.shape[0] == 1, then between each sample of arr1 to arr2
    """
    mask = np.isnan(np.sum(np.vstack((arr1, arr2)), axis=0))
    arr1_m, arr2_m = arr1[:,~mask], arr2[:,~mask]
    vx = arr1_m - arr1_m.mean(axis=-1)[:,None]
    vy = arr2_m - arr2_m.mean(axis=-1)[:,None]
    numerator = np.sum(vx * vy, axis=1)
    denom = np.sqrt(np.sum(vx**2, axis=1)) * np.sqrt(np.sum(vy**2, axis=1))
    return numerator / denom
