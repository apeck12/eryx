import numpy as np

def generate_grid(A_inv, hrange, krange, lrange):
    """
    Generate a grid of q-vectors based on the desired extents 
    and spacing in hkl space.
    
    Parameters
    ----------
    A_inv : numpy.ndarray, shape (3,3)
        fractional cell orthogonalization matrix
    hrange : tuple, shape (3,)
        (hmin, hmax, oversampling relative to Miller indices)
    krange : tuple, shape (3,)
        (kmin, kmax, oversampling relative to Miller indices)
    lrange : tuple, shape (3,)
        (lmin, lmax, oversampling relative to Miller indices)
    
    Returns
    -------
    q_grid : numpy.ndarray, shape (n_points, 3)
        grid of q-vectors
    map_shape : tuple, shape (3,)
        shape of 3d map
    """
    hsteps = hrange[2]*(hrange[1]-hrange[0])+1
    ksteps = krange[2]*(krange[1]-krange[0])+1
    lsteps = lrange[2]*(lrange[1]-lrange[0])+1
    map_shape = (hsteps, ksteps, lsteps)
    
    hkl_grid = np.mgrid[lrange[0]:lrange[1]:lsteps*1j,
                        krange[0]:krange[1]:ksteps*1j,
                        hrange[0]:hrange[1]:hsteps*1j]
    hkl_grid = hkl_grid.T.reshape(-1,3)
    hkl_grid = hkl_grid[:, [2,1,0]]
    
    q_grid = 2*np.pi*np.inner(A_inv, hkl_grid).T
    return q_grid, map_shape
