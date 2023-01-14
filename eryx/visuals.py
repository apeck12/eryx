import matplotlib.pyplot as plt
import numpy as np

def visualize_central_slices(I, vmax_scale=5):
    """
    Plot central slices from the input map,  assuming
    that the map is centered around h,k,l=(0,0,0).

    Parameters
    ----------
    I : numpy.ndarray, 3d
        intensity map
    vmax_scale : float
        vmax will be vmax_scale*mean(I)
    """
    f, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12,4))
    map_shape = I.shape
    vmax = I[~np.isnan(I)].mean()*vmax_scale
    
    ax1.imshow(I[int(map_shape[0]/2),:,:], vmax=vmax)
    ax2.imshow(I[:,int(map_shape[1]/2),:], vmax=vmax)
    ax3.imshow(I[:,:,int(map_shape[2]/2)], vmax=vmax)

    ax1.set_aspect(map_shape[2]/map_shape[1])
    ax2.set_aspect(map_shape[2]/map_shape[0])
    ax3.set_aspect(map_shape[1]/map_shape[0])

    ax1.set_title("(0,k,l)", fontsize=14)
    ax2.set_title("(h,0,l)", fontsize=14)
    ax3.set_title("(h,k,0)", fontsize=14)

    for ax in [ax1,ax2,ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
