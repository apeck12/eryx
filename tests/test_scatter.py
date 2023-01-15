import numpy as np
import gemmi
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.pdb import AtomicModel
import eryx.reference as reference
import eryx.scatter as scatter
import eryx.map_utils as map_utils

class TestScatter(object):
    """
    Check faster implementations of the scattering calculations.
    """
    @classmethod
    def setup_class(cls):
        cls.pdb_path = "pdbs/histidine_p1.pdb"
        cls.model = AtomicModel(cls.pdb_path, clean_pdb=False)
        cls.model.flatten_model()
        cls.q_grid, cls.map_shape = map_utils.generate_grid(cls.model.A_inv, (-2,2,5), (-2,2,5), (-2,2,5))
        
    def test_compute_form_factors(self):
        """ Check that the form factors calculation is correct. """
        indices = np.random.randint(0, high=self.q_grid.shape[0], size=4)
        atom_index = np.random.randint(0, high=self.model.ff_a.shape[0], size=1)[0]
        fj = scatter.compute_form_factors(self.q_grid[indices], self.model.ff_a, self.model.ff_b, self.model.ff_c)[:,atom_index]
        stols2 = np.square(np.linalg.norm(self.q_grid[indices], axis=1) / (4*np.pi))
        ref_fj = np.array([self.model.elements[atom_index].it92.calculate_sf(st2) for st2 in stols2])
        assert np.allclose(ref_fj, fj)

    def test_structure_factors(self):
        """ Check that accelerated structure factors calculation is correct. """
        U = np.random.randn(self.model.xyz.shape[0])
        sf_np8 = scatter.structure_factors(self.q_grid, self.model.xyz, self.model.ff_a, self.model.ff_b, self.model.ff_c, U)
        sf_np1 = scatter.structure_factors(self.q_grid, self.model.xyz, self.model.ff_a, self.model.ff_b, self.model.ff_c, U, n_processes=1)
        sf_ref = reference.structure_factors(self.q_grid, self.model.xyz, self.model.elements, U)
        assert np.allclose(np.square(np.abs(sf_np8)), np.square(np.abs(sf_ref)))
        assert np.allclose(np.square(np.abs(sf_np1)), np.square(np.abs(sf_ref)))
        
    def test_structure_factors_vs_gemmi(self):
        """ Check that structure factors calculation matches gemmi. """
        hkl = np.random.randint(-10, high=10, size=3)
        structure = gemmi.read_pdb(self.pdb_path)
        calc_x = gemmi.StructureFactorCalculatorX(structure.cell)
        sf_ref = np.array(calc_x.calculate_sf_from_model(structure[0], hkl))
        A_inv = np.array(structure.cell.fractionalization_matrix)
        q_vec = 2*np.pi*np.inner(A_inv.T, np.array(hkl))
        sf = scatter.structure_factors(np.array([q_vec]), self.model.xyz, self.model.ff_a, self.model.ff_b, self.model.ff_c, U=None)
        assert np.allclose(sf, sf_ref)
        
class TestReference(object):
    """
    Check reference implementation of the diffuse calculation from a covariance matrix.
    """
    @classmethod
    def setup_class(cls):
        cls.model = AtomicModel("pdbs/pentagon.pdb", expand_p1=False)
        cls.model.flatten_model()
        cls.q_grid, cls.map_shape = map_utils.generate_grid(cls.model.A_inv, (-1,1,9), (-1,1,9), (-1,1,9))
        
    def test_no_disorder(self):
        """ Check Bragg and diffuse components without disorder. """
        Viso = np.zeros((self.model.xyz.shape[0], self.model.xyz.shape[0]))
        Vaniso = np.zeros((self.model.xyz.shape[0], self.model.xyz.shape[0], 3, 3))
        Id_iso, Ib_iso = reference.diffuse_covmat(self.q_grid, self.model.xyz, self.model.elements, Viso)
        Id_aniso, Ib_aniso = reference.diffuse_covmat(self.q_grid, self.model.xyz, self.model.elements, Vaniso)
        Ib_ref = np.square(np.abs(reference.structure_factors(self.q_grid, self.model.xyz, self.model.elements, np.zeros(self.model.xyz.shape[0]))))
        # bragg components should match
        assert np.allclose(Ib_ref, Ib_iso)
        assert np.allclose(Ib_ref, Ib_aniso)
        # diffuse component should be zero
        assert np.allclose(0, Id_iso)
        assert np.allclose(0, Id_aniso)

    def test_uncorrelated_disorder(self):
        """ Check case of uncorrelated disorder. """
        adps = np.random.randn(self.model.xyz.shape[0]) # (n_atoms)
        adps_iso = np.diagflat(adps) # (n_atoms, n_atoms)
        adps_aniso = adps_iso[:,:,None,None] * np.eye(3)[None,None,:] # (n_atoms, n_atoms, 3, 3)
        Ib_ref = np.square(np.abs(reference.structure_factors(self.q_grid, self.model.xyz, self.model.elements, adps)))
        Id_iso, Ib_iso = reference.diffuse_covmat(self.q_grid, self.model.xyz, self.model.elements, adps_iso)
        Id_aniso, Ib_aniso = reference.diffuse_covmat(self.q_grid, self.model.xyz, self.model.elements, adps_aniso)
        # bragg components match
        assert np.allclose(Ib_ref, Ib_iso)
        assert np.allclose(Ib_ref, Ib_aniso)
        # diffuse components match and are radially symmetric
        assert np.allclose(Id_iso, Id_aniso)
        assert np.allclose(Id_iso.reshape(self.map_shape)[10,:,:], Id_iso.reshape(self.map_shape)[:,10,:])
        assert np.allclose(Id_aniso.reshape(self.map_shape)[:,:,10], Id_aniso.reshape(self.map_shape)[:,10,:])
        
