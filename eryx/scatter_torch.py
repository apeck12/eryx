import numpy as np
import torch
import ray
import os
import psutil
import multiprocess as mp
from functools import partial

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
    fj : torch.Tensor, shape (n_points, n_atoms)
        atomic form factors 
    """
    Q = (torch.sqrt(torch.Tensor(q_grid).pow(2).sum(1)) / (4 * np.pi)).pow(2)
    fj = torch.exp(-1 * torch.Tensor(ff_b).unsqueeze_(-1) * Q.unsqueeze(0))
    fj *= torch.Tensor(ff_a).unsqueeze_(-1)
    fj = fj.sum(1) + torch.Tensor(ff_c).unsqueeze(-1)
    return torch.transpose(fj, 0, 1)

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
        U = torch.zeros(xyz.shape[0])
    U = torch.Tensor(U)

    q_grid = torch.Tensor(q_grid)
    qmags = torch.sqrt(q_grid.pow(2).sum(1))
    dwf = torch.exp(-0.5 * torch.square(qmags.unsqueeze(-1)) * U)
    qr = torch.matmul(q_grid, torch.Tensor(xyz).T)

    fj = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    A = 1j * fj * torch.sin(qr) * dwf
    A += fj * torch.cos(qr) * dwf
    
    if compute_qF:
        A = A.unsqueeze(-1) * q_grid.unsqueeze(1)
        A = torch.reshape(A, (A.shape[0], -1))    
    if project_on_components is not None:
        A = torch.matmul(A, torch.Tensor(project_on_components).to(torch.complex64))
    if sum_over_atoms:
        A = torch.sum(A, 1)
        
    return A

def structure_factors_torch(q_grid, xyz, ff_a, ff_b, ff_c, U=None,
                      batch_size=100000, n_processes=8,
                      compute_qF=False, project_on_components=None,
                      sum_over_atoms=True):
    """
    Batched version of the structure factor calculation. See 
    docstring for structure_factors_batch for parameters and 
    returns, with the exception of n_processes, which refers 
    to the number of processors available. If greater than 1,
    multiprocessing will be used.
    """

    n_batches = q_grid.shape[0] // batch_size
    if n_batches == 0:
        n_batches = 1
    splits = np.append(np.arange(n_batches) * batch_size, np.array([q_grid.shape[0]]))

    if n_processes == 1:
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
            A[splits[batch]: splits[batch+1]] = structure_factors_batch(q_sel, 
                                                                        xyz, 
                                                                        ff_a, 
                                                                        ff_b, 
                                                                        ff_c, 
                                                                        U=U, 
                                                                        compute_qF=compute_qF,
                                                                        project_on_components=project_on_components, 
                                                                        sum_over_atoms=sum_over_atoms)
    else:
        q_sel = [q_grid[splits[batch]: splits[batch+1]] for batch in range(n_batches)]
        pool = mp.Pool(processes=n_processes)
        sf_partial = partial(structure_factors_batch, 
                             xyz=xyz, 
                             ff_a=ff_a, 
                             ff_b=ff_b, 
                             ff_c=ff_c, U=U,
                             compute_qF=compute_qF, 
                             project_on_components=project_on_components, 
                             sum_over_atoms=sum_over_atoms)
        A = np.concatenate(list(pool.imap(sf_partial, q_sel)), axis=0)
        pool.close()
        pool.join()
        
    return A

def structure_factors(q_grid, xyz, ff_a, ff_b, ff_c, U=None,
                      batch_size=100000, parallelize='multiprocess',
                      compute_qF=False, project_on_components=None,
                      sum_over_atoms=True):
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
