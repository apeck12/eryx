import numpy as np

def diffuse_covmat(q_grid, xyz, elements, V):
    """
    Compute a diffuse scattering map for disorder in the harmonic 
    approximation, i.e. from a interatomic covariance matrix.
    
    Parameters
    ----------
    q_grid : numpy.ndarray, shape (n_points, 3)
        q-vectors in Angstrom
    xyz : numpy.ndarray, shape (n_atoms, 3)
        atomic xyz positions in Angstroms
    elements : list of gemmi.Element objects
        element objects, ordered as xyz
    V : numpy.ndarray, shape (n_atoms, n_atoms, 3, 3) or (n_atoms, n_atoms)
        covariance matrix of interatomic displacements
        
    Returns
    -------
    Id : numpy.ndarray, shape (n_points)
        diffuse scattering at q-vectors
    Ib : numpy.ndarray, shape (n_points)
        Bragg scattering at q-vectors
    """
    
    Id = np.zeros(q_grid.shape[0])  
    Ib = np.zeros(q_grid.shape[0])  
    stols2 = np.square(np.linalg.norm(q_grid, axis=1) / (4*np.pi)) 
    
    for i,q_vector in enumerate(q_grid):
        Fd, Fb = 0.0, 0.0
        
        for j in range(xyz.shape[0]):
            for k in range(xyz.shape[0]):
                
                fj = elements[j].it92.calculate_sf(stols2[i])
                fk = elements[k].it92.calculate_sf(stols2[i])
                rjk = xyz[j] - xyz[k]
                
                qVjjq = np.dot(q_vector, np.dot(V[j][j], q_vector))
                qVkkq = np.dot(q_vector, np.dot(V[k][k], q_vector))
                qVjkq = np.dot(q_vector, np.dot(V[j][k], q_vector))
                
                Fb += fj * fk * np.exp(-1j * np.dot(q_vector, rjk)) * np.exp(-0.5 * qVjjq - 0.5 * qVkkq) 
                Fd += fj * fk * np.exp(-1j * np.dot(q_vector, rjk)) * np.exp(-0.5 * qVjjq - 0.5 * qVkkq) * (np.exp(qVjkq) - 1)
        
        Id[i], Ib[i] = Fd.real, Fb.real
        
    return Id, Ib

def structure_factors(q_grid, xyz, elements, adps, sf_complex=False):
    """
    Compute the structure factor intensities for an atomic model at 
    the given q-vectors. 

    Parameters
    ----------
    q_grid : numpy.ndarray, shape (n_points, 3)
        q-vectors in Angstrom
    xyz : numpy.ndarray, shape (n_atoms, 3)
        atomic xyz positions in Angstroms
    elements : list of gemmi.Element objects
        element objects, ordered as xyz
    adps : numpy.ndarray, shape (n_atoms) or (n_atoms, 3, 3)
        (an)isotropic displacement parameters
    sf_complex : bool
        if False, return intensities rather than complex values
        
    Returns
    -------
    sf : numpy.ndarray, shape (n_points)
        structure factors at q-vectors
    """
    
    sf = np.zeros(q_grid.shape[0], dtype=complex)
    stols2 = np.square(np.linalg.norm(q_grid, axis=1) / (4*np.pi)) 
    
    for i,q_vector in enumerate(q_grid):
        F = 0.0
        
        for j in range(xyz.shape[0]):
            fj = elements[j].it92.calculate_sf(stols2[i])
            qVjq = np.dot(q_vector, np.dot(adps[j], q_vector))
            F += fj * np.exp(-1j * np.dot(q_vector, xyz[j])) * np.exp(-0.5 * qVjq)
            
        sf[i] = F
    
    if not sf_complex:
        sf = np.square(np.abs(sf))

    return sf
