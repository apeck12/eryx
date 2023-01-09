import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.pdb import AtomicModel
import eryx.map_utils as map_utils

def setup_model(case):
    """ Return variables for a test case in the given space group. """
    
    if case == 'orthorhombic':
        hsampling = (-5,5,3)
        ksampling = (-17,17,2)
        lsampling = (-30,30,2)
        pdb_path = "/Users/apeck/Desktop/diffuse/eryx/tests/pdbs/5zck.pdb"

    elif case == 'trigonal':
        hsampling = (-27,27,2)
        ksampling = (-27,27,2)
        lsampling = (-5,5,3)
        pdb_path = "/Users/apeck/Desktop/diffuse/eryx/tests/pdbs/7n2h.pdb"

    elif case == 'triclinic':
        hsampling = (-14, 14, 2)
        ksampling = (-5, 5, 2)
        lsampling = (-15, 15, 2)
        pdb_path = "/Users/apeck/Desktop/diffuse/eryx/tests/pdbs/2ol9.pdb"
        
    else:
        raise ValueError("Currently only orthorhombic, trigonal, and triclinic are supported.")
    
    model = AtomicModel(pdb_path, expand_p1=False)
    return model, hsampling, ksampling, lsampling

def test_compute_multiplicity():
    """ Test that the correct multiplicity values are assigned to hkl indices. """

    for case in ['orthorhombic', 'triclinic']:
        model, hsampling, ksampling, lsampling = setup_model(case)
        hkl_grid, multiplicity = map_utils.compute_multiplicity(model, hsampling, ksampling, lsampling)
        zeros = 3 - np.count_nonzero(hkl_grid, axis=1)
        
        if case == 'orthorhombic':
            assert np.allclose(multiplicity.flatten()[zeros==0], 8)
            assert np.allclose(multiplicity.flatten()[zeros==1], 4)
            assert np.allclose(multiplicity.flatten()[zeros==2], 2)
            
        if case == 'triclinic':
            assert np.allclose(multiplicity.flatten()[zeros!=3], 2)
