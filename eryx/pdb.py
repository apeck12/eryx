import numpy as np
import gemmi

"""
Functions to handle coordinates files using the gemmi library.
"""

def get_xyz_asus(xyz, structure):
    """
    Apply symmetry operations to get xyz coordinates
    for all asymmetric units in the unit cell.
    
    Parameters
    ----------
    xyz : numpy.ndarray, shape (n_atoms, 3)
        atomic coordinates for a single asymmetric unit
    structure : gemmi.Structure 
        gemmi object with space group and symmetry info

    Returns
    -------
    xyz_asus : numpy.ndarray, shape (n_asu, n_atoms, 3)
        atomic coordinates for all asus in the unit cell
    """
    sg = gemmi.SpaceGroup(structure.spacegroup_hm)
    cell = np.array(structure.cell.parameters)

    xyz_asus = []
    for op in sg.operations():
        rot = np.array(op.rot) / op.DEN
        trans = np.array(op.tran) / op.DEN 
        xyz_asu = np.inner(rot, xyz).T + trans * cell[:3]  
        com = np.mean(xyz_asu.T, axis=1)
        xyz_asu += cell[:3] * np.array(com < 0).astype(int)
        xyz_asus.append(xyz_asu)
    return np.array(xyz_asus)

def extract_xyz(pdb_file, frame=0, expand_p1=False):
    """
    Extract atomic coordinates, optionally retrieving xyz
    positions for each asu in the unit cell.

    Returns
    -------
    xyz : numpy.ndarray, shape (n_asu or 1, n_atoms, 3)
        atomic coordinates in Angstroms
    """
    structure = gemmi.read_pdb(pdb_file)
    model = structure[frame]
    residues = [res for ch in model for res in ch]
    xyz = np.array([atom.pos.tolist() for res in residues for atom in res])

    if expand_p1:
        xyz = get_xyz_asus(xyz, structure)
    else:
        xyz = np.array([xyz])

    return xyz
        
def extract_model_info(pdb_file, frame=0):
    """
    Extract the atomic coordinates and element information 
    from an atomic coordinates file.
    
    Parameters
    ----------
    pdb_file : str
        coordinates file in PDB format
    frame : int
        index of frame to retrieve

    Returns
    -------
    xyz : numpy.ndarray, shape (n_atoms, 3) 
        atomic coordinates in Angstroms
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

