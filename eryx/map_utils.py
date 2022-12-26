import matplotlib.pyplot as plt
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
    hsteps = int(hrange[2]*(hrange[1]-hrange[0])+1)
    ksteps = int(krange[2]*(krange[1]-krange[0])+1)
    lsteps = int(lrange[2]*(lrange[1]-lrange[0])+1)
    
    hkl_grid = np.mgrid[lrange[0]:lrange[1]:lsteps*1j,
                        krange[0]:krange[1]:ksteps*1j,
                        hrange[0]:hrange[1]:hsteps*1j]
    map_shape = hkl_grid.shape[1:][::-1]
    hkl_grid = hkl_grid.T.reshape(-1,3)
    hkl_grid = hkl_grid[:, [2,1,0]]
    
    q_grid = 2*np.pi*np.inner(A_inv, hkl_grid).T
    return q_grid, map_shape

def visualize_central_slices(I, vmax_scale=5):
    """
    Plot input map's central slices, assuming that map
    is centered around h,k,l=(0,0,0).

    Parameters
    ----------
    I : numpy.ndarray, 3d
        intensity map
    vmax_scale : float
        vmax will be vmax_scale*mean(I)
    """
    f, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12,4))
    map_shape = I.shape
    
    ax1.imshow(I[int(map_shape[0]/2),:,:], vmax=I.mean()*vmax_scale)
    ax2.imshow(I[:,int(map_shape[1]/2),:], vmax=I.mean()*vmax_scale)
    ax3.imshow(I[:,:,int(map_shape[2]/2)], vmax=I.mean()*vmax_scale)

    ax1.set_aspect(map_shape[2]/map_shape[1])
    ax2.set_aspect(map_shape[2]/map_shape[0])
    ax3.set_aspect(map_shape[1]/map_shape[0])

    ax1.set_title("(0,k,l)", fontsize=14)
    ax2.set_title("(h,0,l)", fontsize=14)
    ax3.set_title("(h,k,0)", fontsize=14)

    for ax in [ax1,ax2,ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
