import numpy as np
from .pdb import AtomicModel
from .map_utils import generate_grid
from .scatter import structure_factors

def compute_transform(transform, pdb_path, hsampling, ksampling, lsampling, U=None, batch_size=10000):
    """
    Compute either the crystal transform or the molecular transform
    as the coherent and incoherent sums of the scattering from each 
    asymmetric unit, respectively.
    
    Parameters
    ----------
    transform : str
        molecular or crystal
    pdb_path : str
        path to coordinates file of asymmetric unit
    hsampling : tuple, shape (3,)
        (hmin, hmax, oversampling relative to Miller indices)
    ksampling : tuple, shape (3,)
        (kmin, kmax, oversampling relative to Miller indices)
    lsampling : tuple, shape (3,)
        (lmin, lmax, oversampling relative to Miller indices)
    U : numpy.ndarray, shape (n_atoms,)
        isotropic displacement parameters, applied to each asymmetric unit
    batch_size : int
        number of q-vectors to evaluate per batch
        
    Returns
    -------
    q_grid : numpy.ndarray, (n_points, 3)
        q-vectors corresponding to flattened intensity map
    I : numpy.ndarray, 3d
        intensity map of the molecular transform
    """
    if transform=='molecular':
        model = AtomicModel(pdb_path, expand_p1=True)
    elif transform=='crystal':
        model = AtomicModel(pdb_path)
        model.xyz = np.expand_dims(model.xyz, axis=0)
    else:
        raise ValueError("Transform type not recognized. Must be molecular or crystal")
        
    q_grid, map_shape = generate_grid(model.A_inv, hsampling, ksampling, lsampling)
    I = np.zeros(q_grid.shape[0])
    for asu in range(model.xyz.shape[0]):
        A = structure_factors(q_grid, model.xyz[asu], model.ff_a, model.ff_b, model.ff_c, U=U, batch_size=batch_size)
        I += np.square(np.abs(A))
    return q_grid, I.reshape(map_shape)
