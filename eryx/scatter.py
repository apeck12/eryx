import numpy as np

def compute_form_factors(q_grid, ff_a, ff_b, ff_c):
    """
    Compute atomic form factors for the input q-vectors.
    
    Parameters
    ----------
    q_grid : numpy.ndarray, shape (n_points, 3)
        q-vectors in Angstrom
    ff_a : numpy.ndarray, shape (n_atoms, 4)
        a coefficient of atomic form factors
    ff_b : numpy.ndarray, shape (n_atoms, 4)
        b coefficient of atomic form factors
    ff_c : numpy.ndarray, shape (n_atoms,)
        c coefficient of atomic form factors

    Returns
    -------
    fj : numpy.ndarray, shape (n_points, n_atoms)
        atomic form factors 
    """
    Q = np.square(np.linalg.norm(q_grid, axis=1) / (4*np.pi))
    fj = ff_a[:,:,np.newaxis] * np.exp(-1 * ff_b[:,:,None] * Q[:,np.newaxis].T)
    fj = np.sum(fj, axis=1) + ff_c[:,np.newaxis]
    return fj.T
