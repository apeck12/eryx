import numpy as np

def compute_cc(arr1, arr2, mask=None):
    """
    Compute the Pearson correlation coefficient between the input arrays.
    Voxels that should be ignored are assumed to have a value of NaN.
    
    Parameters
    ----------
    arr1 : numpy.ndarray, shape (n_samples, n_points)
        input array
    arr2 : numpy.ndarray, shape (n_samples, n_points) or (1, n_points)
        input array to compute CC with
    mask : numpy.ndarray, shape (map_shape) or (n_points,)
        e.g. to select asu/resolution. True values indicate retained grid points
    
    Returns
    -------
    ccs : numpy.ndarray, shape (n_samples)
        correlation coefficient between paired sample arrays, or if
        arr2.shape[0] == 1, then between each sample of arr1 to arr2
    """
    if len(arr1.shape) == 1:
        arr1 = np.array([arr1])
    if len(arr2.shape) == 1:
        arr2 = np.array([arr2])
    
    valid = ~np.isnan(np.sum(np.vstack((arr1, arr2)), axis=0))
    if mask is not None:
        valid *= mask.flatten()
    arr1_m, arr2_m = arr1[:,valid], arr2[:,valid]
    vx = arr1_m - arr1_m.mean(axis=-1)[:,None]
    vy = arr2_m - arr2_m.mean(axis=-1)[:,None]
    numerator = np.sum(vx * vy, axis=1)
    denom = np.sqrt(np.sum(vx**2, axis=1)) * np.sqrt(np.sum(vy**2, axis=1))
    return numerator / denom

def compute_cc_by_shell(arr1, arr2, res_map, mask=None, n_shells=10):
    """
    Compute the Pearson correlation coefficient by resolution shell between the 
    input arrays. The bin widths of resolutions shells are uniform in d^-3. 
    
    Parameters
    ----------
    arr1 : numpy.ndarray, shape (n_points,)
        input array
    arr2 : numpy.ndarray, shape (n_points,) 
        input array to compute CC with
    res_map : numpy.ndarray, shape (n_points,) 
        resolution in Angstrom of each reciprocal grid point
    mask : numpy.ndarray, shape (map_shape) or (n_points,)
        e.g. to select asu/resolution. True values indicate retained grid points
    n_shells : int
        number of resolution bins
    
    Returns
    -------
    res_shell : numpy.ndarray, shape (n_shells,)
        median resolution of each resolution shell
    cc_shell : numpy.ndarray, shape (n_shells,)
        correlation coefficient by resolution shell
    """
    arr1, arr2 = arr1.flatten(), arr2.flatten()
    if mask is None:
        mask = np.ones(res_map.shape).astype(bool)
    mask *= ~np.isnan(np.sum(np.vstack((arr1, arr2)), axis=0))
        
    inv_dcubed = 1.0 / (res_map ** 3.0)
    res_limit = res_map[mask].min()
    hist, bin_edges = np.histogram(inv_dcubed[res_map>res_limit], bins=n_shells)
    ind = np.digitize(inv_dcubed, bin_edges)
    
    cc_shell, res_shell = np.zeros(n_shells), np.zeros(n_shells)
    for i in range(1, n_shells+1):
        arr1_sel, arr2_sel, mask_sel = arr1[ind==i], arr2[ind==i], mask[ind==i]
        cc_shell[i-1] = compute_cc(arr1_sel, arr2_sel, mask=mask_sel)[0]
        res_shell[i-1] = np.median(res_map[ind==i])
        
    return res_shell, cc_shell

def compute_cc_by_dq(arr1, arr2, dq_map, mask=None):
    """
    Compute the Pearson correlation coefficient as a function of dq, the
    distance between reciprocal grid points and the nearest Bragg peak.
    
    Parameters
    ----------
    arr1 : numpy.ndarray, shape (n_points,)
        input array
    arr2 : numpy.ndarray, shape (n_points,) 
        input array to compute CC with
    dq_map : numpy.ndarray, shape (n_points,)
        distance of each grid point to 
    mask : numpy.ndarray, shape (map_shape) or (n_points,)
        e.g. to select asu/resolution. True values indicate retained grid points
    
    Returns
    -------
    dq_vals : numpy.ndarray, shape (n_unique_dq,)
        unique distances from the nearest Bragg peak
    cc_dq : numpy.ndarray, shape (n_shells,)
        correlation coefficient by resolution shell
    """
    arr1, arr2 = arr1.flatten(), arr2.flatten()
    if mask is None:
        mask = np.ones(dq_map.shape).astype(bool)
    mask *= ~np.isnan(np.sum(np.vstack((arr1, arr2)), axis=0))
    
    dq_vals = np.unique(dq_map)
    cc_dq = np.zeros(len(dq_vals))
    for i,dq in enumerate(dq_vals):
        arr1_sel, arr2_sel, mask_sel = arr1[dq_map==dq], arr2[dq_map==dq], mask[dq_map==dq]
        cc_dq[i] = compute_cc(arr1_sel, arr2_sel, mask=mask_sel)[0]
    
    return dq_vals, cc_dq
