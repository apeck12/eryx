import numpy as np

def structure_factors(q_grid, xyz, elements, U=None):
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
    U : numpy.ndarray, shape (n_atoms) or (n_atoms, 3, 3)
        (an)isotropic displacement parameters
        
    Returns
    -------
    A : numpy.ndarray, shape (n_points)
        complex structure factor amplitudes at q-vectors
    """
    
    A = np.zeros(q_grid.shape[0], dtype=np.complex128)
    stols2 = np.square(np.linalg.norm(q_grid, axis=1) / (4*np.pi)) 
    
    if U is None:
        U = np.zeros(xyz.shape[0])
    
    for i,q_vector in enumerate(q_grid):    
        
        for j in range(xyz.shape[0]):
            q_mag = np.linalg.norm(q_vector)
            fj = elements[j].it92.calculate_sf(stols2[i])
            rj = xyz[j,:]

            if len(U.shape) == 1:
                qUq = np.square(q_mag)*U[j]
            else:
                qUq = np.dot(np.dot(q_vector, U[j]), q_vector)

            A[i] +=      fj * np.sin( np.dot(q_vector, rj) ) * np.exp(- 0.5 * qUq)
            A[i] += 1j * fj * np.cos( np.dot(q_vector, rj) ) * np.exp(- 0.5 * qUq)

    return A

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
        q_mag = np.linalg.norm(q_vector)
        
        for j in range(xyz.shape[0]):
            for k in range(xyz.shape[0]):
                
                fj = elements[j].it92.calculate_sf(stols2[i])
                fk = elements[k].it92.calculate_sf(stols2[i])
                rjk = xyz[j] - xyz[k]
                
                if len(V.shape) == 2:
                    qVjjq = np.square(q_mag)*V[j][j]
                    qVkkq = np.square(q_mag)*V[k][k] 
                    qVjkq = np.square(q_mag)*V[j][k]
                    
                else:
                    qVjjq = np.dot(q_vector, np.dot(V[j][j], q_vector))
                    qVkkq = np.dot(q_vector, np.dot(V[k][k], q_vector))
                    qVjkq = np.dot(q_vector, np.dot(V[j][k], q_vector))
                
                Fb += fj * np.conj(fk) * np.exp(-1j * np.dot(q_vector, rjk)) * np.exp(-0.5 * qVjjq - 0.5 * qVkkq) 
                Fd += fj * np.conj(fk) * np.exp(-1j * np.dot(q_vector, rjk)) * np.exp(-0.5 * qVjjq - 0.5 * qVkkq) * (np.exp(qVjkq) - 1)
        
        Id[i], Ib[i] = Fd.real, Fb.real
        
    return Id, Ib

