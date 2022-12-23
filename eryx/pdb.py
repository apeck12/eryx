import numpy as np
import gemmi

"""
Functions to handle coordinates files using the gemmi library.
"""

def extract_model_info(pdb_file, frame=0):
    """
    Extract the atomic coordinates and element information 
    from an atomic coordinates file.
    
    Parameters
    ----------
    pdb_file : str
        coordinates file in PDB format
        
    Returns
    -------
    xyz : numpy.ndarray, shape (n_atoms, 3)
        atomic xyz positions in Angstroms
    elements : list of gemmi.Element objects
        element objects, ordered as xyz
    """
    structure = gemmi.read_pdb(pdb_file)
    model = structure[frame]
    
    residues = [res for ch in model for res in ch]
    xyz = np.array([atom.pos.tolist() for res in residues for atom in res])
    elements = [atom.element for res in residues for atom in res]
    
    return xyz, elements

def extract_ff_coefs(pdb_file, frame=0):
    """
    Retrieve atomic form factor coefficients, specifically
    a, b, and c of the following equation:
    f(qo) = sum(a_i * exp(-b_i * (qo/(4*pi))^2)) + c
    and the sum is over the element's four coefficients.
    
    Parameters
    ----------
    pdb_file : str
        coordinates file in PDB format
        
    Returns
    -------
    ff_a : numpy.ndarray, shape (n_atoms, 4)
        a coefficient of atomic form factors
    ff_b : numpy.ndarray, shape (n_atoms, 4)
        b coefficient of atomic form factors
    ff_c : numpy.ndarray, shape (n_atoms,)
        c coefficient of atomic form factors
    """
    structure = gemmi.read_pdb(pdb_file)
    model = structure[frame]
    residues = [res for ch in model for res in ch]
    
    ff_a = np.array([atom.element.it92.a for res in residues for atom in res])
    ff_b = np.array([atom.element.it92.b for res in residues for atom in res])
    ff_c = np.array([atom.element.it92.c for res in residues for atom in res])

    return ff_a, ff_b, ff_c
