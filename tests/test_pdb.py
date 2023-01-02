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
    def setup_class(cls):
        cls.pdb_ids = ["5zck", "7n2h", "193l"] # P 21 21 21, P 31, P 43 21 2
        
    def test_various_sg(self):
        """ Check that atomic positions and form factors are properly expanded. """
        for pdb_id in self.pdb_ids:
            # load all frames from Chimera-generated p1 and eryx-expanded asu
            model_p1 = AtomicModel(f"pdbs/{pdb_id}_p1.pdb", frame=-1)
            model = AtomicModel(f"pdbs/{pdb_id}.pdb", expand_p1=True)

            # check that form factors match
            assert np.allclose(model_p1.ff_a, model.ff_a)
            assert np.allclose(model_p1.ff_b, model.ff_b)
            assert np.allclose(model_p1.ff_c, model.ff_c)

            # check that atomic coordinates match within a cell shift
            xyz_ref = np.array([model_p1.xyz[i] for i in range(model_p1.xyz.shape[0])])
            xyz_tar = np.array([model.xyz[i] for i in range(model.xyz.shape[0])])
            xyz_ref = xyz_ref.reshape(-1, xyz_ref.shape[-1])
            xyz_tar = xyz_tar.reshape(-1, xyz_tar.shape[-1])
            shifts = np.mean(np.abs(xyz_ref - xyz_tar), axis=0) / model_p1.cell[:3]
            assert np.allclose(xyz_tar - shifts * model_p1.cell[:3], xyz_ref, atol=1e-2)

    def test_sym_str_as_matrix(self):
        """ Check that the symmetry triplets are correctly converted to matrices. """
        for pdb_id in self.pdb_ids:
            model = AtomicModel(f"pdbs/{pdb_id}.pdb")
            sg = gemmi.SpaceGroup(model.space_group) 
            for i,op in enumerate(sg.operations()):
                converted = sym_str_as_matrix(op.triplet().upper())
                reference = np.array(op.rot)/op.DEN
                assert np.allclose(converted, reference)
