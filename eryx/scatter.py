import numpy as np
import ray
import os
import psutil
import multiprocess as mp
from functools import partial
from tqdm import tqdm

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

def structure_factors_batch(q_grid, xyz, ff_a, ff_b, ff_c, U=None,
                            compute_qF=False, project_on_components=None,
                            sum_over_atoms=True):
    """
    Compute the structure factors for an atomic model at 
    the given q-vectors. 

    Parameters
    ----------
    q_grid : numpy.ndarray, shape (n_points, 3)
        q-vectors in Angstrom
    xyz : numpy.ndarray, shape (n_atoms, 3)
        atomic xyz positions in Angstroms
    ff_a : numpy.ndarray, shape (n_atoms, 4)
        a coefficient of atomic form factors
    ff_b : numpy.ndarray, shape (n_atoms, 4)
        b coefficient of atomic form factors
    ff_c : numpy.ndarray, shape (n_atoms,)
        c coefficient of atomic form factors
    U : numpy.ndarray, shape (n_atoms,) 
        isotropic displacement parameters
    compute_qF : boolean
        False (default).
        If true, return structure factors at q-vectors times q-vectors
    project_on_components : None (default) or numpy.ndarray, shape (n_atoms, n_components)
        Projection matrix to convert structure factors into component factors
    sum_over_atoms: boolean
        True (default) returns summed structure factor.
        
    Returns
    -------
    A : numpy.ndarray, shape (n_points) or (n_points, n_atoms) or (n_points, n_components)
        structure factors at q-vectors.
        If compute_qF, return [q_x A(q), q_y A(q), q_z A(q)] instead of A(q)
    """
    if U is None:
        U = np.zeros(xyz.shape[0])
    
    fj = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    qmags = np.linalg.norm(q_grid, axis=1)
    qUq = np.square(qmags[:,np.newaxis]) * U
    
    A = 1j * fj * np.sin(np.dot(q_grid, xyz.T)) * np.exp(-0.5 * qUq)
    A += fj * np.cos(np.dot(q_grid, xyz.T)) * np.exp(-0.5 * qUq)
    if compute_qF:
        A = A[:,:,None] * q_grid[:,None,:]
        A = A.reshape((A.shape[0],A.shape[1]*A.shape[2]))
    if project_on_components is not None:
        A = np.matmul(A, project_on_components)
    if sum_over_atoms:
        A = np.sum(A, axis=1)
    return A 

def structure_factors(q_grid, xyz, ff_a, ff_b, ff_c, U=None,
                      batch_size=100000, parallelize='multiprocess',
                      compute_qF=False, project_on_components=None,
                      sum_over_atoms=True, progress_bar=True):
    """
    Batched version of the structure factor calculation. See 
    docstring for structure_factors_batch for parameters and 
    returns, with the exception of the parallelize argument,
    which can be None (for 1 core), 'multiprocess', or 'ray'.
    We use psutil to automatically deduce the number of cores
    available.
    """
    n_batches = q_grid.shape[0] // batch_size
    if n_batches == 0:
        n_batches = 1
    splits = np.append(np.arange(n_batches) * batch_size, np.array([q_grid.shape[0]]))

    if parallelize is None:
        dim1_size = q_grid.shape[0]
        if sum_over_atoms:
            A_shape = dim1_size
        else:
            if project_on_components is None:
                dim2_size = xyz.shape[0]
            else:
                dim2_size = project_on_components.shape[1]
            A_shape = (dim1_size, dim2_size)
        A = np.zeros(A_shape, dtype=np.complex128)
        for batch in range(n_batches):
            q_sel = q_grid[splits[batch]: splits[batch+1]]
            A[splits[batch]: splits[batch+1]] = structure_factors_batch(q_sel, xyz, ff_a, ff_b, ff_c, U=U, compute_qF=compute_qF,
                                                                        project_on_components=project_on_components, sum_over_atoms=sum_over_atoms)
    else:
        num_cpus = psutil.cpu_count(logical=False)
        q_sel = [q_grid[splits[batch]: splits[batch+1]] for batch in range(n_batches)]

        if parallelize == 'multiprocess':
            pool = mp.Pool(processes=num_cpus)
            sf_partial = partial(structure_factors_batch, xyz=xyz, ff_a=ff_a, ff_b=ff_b, ff_c=ff_c, U=U,
                                 compute_qF=compute_qF, project_on_components=project_on_components, sum_over_atoms=sum_over_atoms)
            if progress_bar:
                A = np.concatenate(list(tqdm(pool.imap(sf_partial, q_sel), total=len(q_sel))), axis=0)
            else:
                A = np.concatenate(pool.map(sf_partial, q_sel), axis=0)
            
        elif parallelize == 'ray':
            os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ray.init()
            A = ray.get([partial(ray.remote(structure_factors_batch).remote,
                                 xyz=xyz, ff_a=ff_a, ff_b=ff_b, ff_c=ff_c, U=U,
                                 compute_qF=compute_qF,
                                 project_on_components=project_on_components,
                                 sum_over_atoms=sum_over_atoms)(q_grid=q_sel[i]) for i in range(len(q_sel))])
            A = np.concatenate(A, axis=0)
            ray.shutdown()

        else:
            raise ValueError("parallize argument not recognized; must be None, multiprocess, or ray")
       
    return A
