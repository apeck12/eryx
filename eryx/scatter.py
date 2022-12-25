import numpy as np

def compute_form_factors(q_grid, ff_a, ff_b, ff_c):
    """
    Evaluate atomic form factors at the input q-vectors.
    
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

def structure_factors(q_grid, xyz, ff_a, ff_b, ff_c, U=None):
    """
    Compute the structure factors for an atomic model at 
    the given q-vectors. 
​
    Parameters
    ----------
    q_grid : numpy.ndarray, shape (n_points, 3)
        q-vectors in Angstrom
    xyz : numpy.ndarray, shape (n_atoms, 3)
        atomic xyz positions in Angstroms
    elements : list of gemmi.Element objects
        element objects, ordered as xyz
    adps : numpy.ndarray, shape (n_atoms,) 
        isotropic displacement parameters
    sf_complex : bool
        if False, return intensities rather than complex values
        
    Returns
    -------
    A : numpy.ndarray, shape (n_points)
        structure factors at q-vectors
    """
    if U is None:
        U = np.zeros(xyz.shape[0])
    
    fj = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    qmags = np.linalg.norm(q_grid, axis=1)
    qUq = np.square(qmags[:,np.newaxis]) * U
    
    A = 1j * fj * np.sin(np.dot(q_grid, xyz.T)) * np.exp(-0.5 * qUq)
    A += fj * np.cos(np.dot(q_grid, xyz.T)) * np.exp(-0.5 * qUq)
    A = np.sum(A, axis=1)
    return A 
