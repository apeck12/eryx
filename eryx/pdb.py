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
