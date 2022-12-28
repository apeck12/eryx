import matplotlib.pyplot as plt
import numpy as np

def generate_grid(A_inv, hsampling, ksampling, lsampling):
    """
    Generate a grid of q-vectors based on the desired extents 
    and spacing in hkl space.
    
    Parameters
    ----------
    A_inv : numpy.ndarray, shape (3,3)
        fractional cell orthogonalization matrix
    hsampling : tuple, shape (3,)
        (hmin, hmax, oversampling relative to Miller indices)
    ksampling : tuple, shape (3,)
        (kmin, kmax, oversampling relative to Miller indices)
    lsampling : tuple, shape (3,)
        (lmin, lmax, oversampling relative to Miller indices)
    
    Returns
    -------
    q_grid : numpy.ndarray, shape (n_points, 3)
        grid of q-vectors
    map_shape : tuple, shape (3,)
        shape of 3d map
    """
    hsteps = int(hsampling[2]*(hsampling[1]-hsampling[0])+1)
    ksteps = int(ksampling[2]*(ksampling[1]-ksampling[0])+1)
    lsteps = int(lsampling[2]*(lsampling[1]-lsampling[0])+1)
    
    hkl_grid = np.mgrid[lsampling[0]:lsampling[1]:lsteps*1j,
                        ksampling[0]:ksampling[1]:ksteps*1j,
                        hsampling[0]:hsampling[1]:hsteps*1j]
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
