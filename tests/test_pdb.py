import numpy as np
import gemmi
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.pdb import *

class TestPDB(object):
    """
    Check correctness of PDB extraction methods.
    """
    @classmethod
    def setup_class(cls):
        cls.pdb_p212121 = "histidine.pdb"
        cls.pdb_p1 = "histidine_p1.pdb"
        cls.n_asu = 4
        
    def test_get_xyz_asus(self):
        """ Check that asu coordinates are correctly calculated. """
        xyz_p1 = extract_xyz(self.pdb_p1)[0]
        asu = {i:xyz_p1[i*21:i*21+21,:] for i in range(self.n_asu)}
        xyz = extract_xyz(self.pdb_p212121, expand_p1=True)
        match = list()
        for i in range(xyz.shape[0]):
            arr = np.array([np.sum(xyz[i] - asu[j]) for j in asu.keys()])
            match.append(np.where(np.around(arr, 6)==0)[0][0])
        match = np.sort(np.array(match))
        assert np.allclose(match, np.arange(xyz.shape[0]))
