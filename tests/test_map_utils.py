import numpy as np
import gemmi
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.pdb import AtomicModel
import eryx.map_utils as map_utils
from base import setup_model

def test_compute_multiplicity():
    """ Test that the correct multiplicity values are assigned to hkl indices. """

    for case in ['orthorhombic', 'triclinic']:
        pdb_path, model, hsampling, ksampling, lsampling = setup_model(case)
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
        pdb_path, model, hsampling, ksampling, lsampling = setup_model(case)
        hkl_grid, map_shape = map_utils.generate_grid(model.A_inv, (-1,1,1), (-1,1,1), (-1,1,1), return_hkl=True)
        mask = map_utils.get_asu_mask(model.space_group, hkl_grid)
        hkl_asu = hkl_grid[mask].astype(int)

        asu = gemmi.ReciprocalAsu(gemmi.SpaceGroup(model.space_group))
        assert all([asu.is_in(list(h)) for h in hkl_asu])

def test_get_dq_map():
    """ Check that the unique dq match their expected values. """
    for case in ['orthorhombic', 'tetragonal']:
        pdb_path, model, hsampling, ksampling, lsampling = setup_model(case)
        if case == 'tetragonal':
            hsampling, ksampling = (-30, 30, 2), (-30, 30, 2) # can't have integral sampling
        hkl_grid, map_shape = map_utils.generate_grid(model.A_inv, hsampling, ksampling, lsampling, return_hkl=True)
        dq_calc = np.unique(map_utils.get_dq_map(model.A_inv, hkl_grid))
        dq_exp = 2*np.pi*np.linalg.norm(model.A_inv.T, axis=0) 
        dq_exp /= np.array([hsampling[2], ksampling[2], lsampling[2]])
        dq_exp = np.append(dq_exp, np.array([np.sqrt(np.square(dq_exp[0]) + np.square(dq_exp[1])), 
                                             np.sqrt(np.square(dq_exp[0]) + np.square(dq_exp[2])),
                                             np.sqrt(np.square(dq_exp[1]) + np.square(dq_exp[2])),
                                             np.sqrt(np.sum(np.square(dq_exp)))]))
        dq_exp = np.append(np.zeros(1), dq_exp)
        dq_exp = np.unique(np.around(dq_exp, decimals=8))
        assert np.allclose(dq_exp, dq_calc)
