import numpy as np
import gemmi
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
        pdb_path = "pdbs/5zck.pdb"

    elif case == 'trigonal':
        hsampling = (-27,27,2)
        ksampling = (-27,27,2)
        lsampling = (-5,5,3)
        pdb_path = "pdbs/7n2h.pdb"

    elif case == 'triclinic':
        hsampling = (-14, 14, 2)
        ksampling = (-5, 5, 2)
        lsampling = (-15, 15, 2)
        pdb_path = "pdbs/2ol9.pdb"
        
    elif case == 'tetragonal':
        hsampling = (-30, 30, 1)
        ksampling = (-30, 30, 1)
        lsampling = (-16, 16, 1)
        pdb_path = "pdbs/193l.pdb"   

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

def test_get_asu_mask():
    """ Test that reflections belonging to the asymmetric unit are correctly identified. """

    for case in ['orthorhombic', 'triclinic', 'orthorhombic', 'trigonal']:
        model, hsampling, ksampling, lsampling = setup_model(case)
        hkl_grid, map_shape = map_utils.generate_grid(model.A_inv, (-1,1,1), (-1,1,1), (-1,1,1), return_hkl=True)
        mask = map_utils.get_asu_mask(model.space_group, hkl_grid)
        hkl_asu = hkl_grid[mask].astype(int)

        asu = gemmi.ReciprocalAsu(gemmi.SpaceGroup(model.space_group))
        assert all([asu.is_in(list(h)) for h in hkl_asu])
