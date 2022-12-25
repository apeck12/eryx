import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.pdb import *
import eryx.reference as reference
import eryx.scatter as scatter

class TestScatter(object):
    """
    Check faster implementations of the scattering calculations.
    """
    @classmethod
    def setup_class(cls):
        qmesh = np.mgrid[-2:2:21j,-2:2:21j,-2:2:21j]
        cls.map_shape = qmesh.shape[1:]
        cls.q_grid = qmesh.T.reshape(-1, 3)
        cls.xyz, cls.elements = extract_model_info("histidine.pdb")
        cls.ff_a, cls.ff_b, cls.ff_c = extract_ff_coefs("histidine.pdb")
        
    def test_compute_form_factors(self):
        """ Check that the form factors calculation is correct. """
        indices = np.random.randint(0, high=self.q_grid.shape[0], size=4)
        atom_index = np.random.randint(0, high=self.ff_a.shape[0], size=1)[0]
        fj = scatter.compute_form_factors(self.q_grid[indices], self.ff_a, self.ff_b, self.ff_c)[:,atom_index]
        stols2 = np.square(np.linalg.norm(self.q_grid[indices], axis=1) / (4*np.pi))
        ref_fj = np.array([self.elements[atom_index].it92.calculate_sf(st2) for st2 in stols2])
        assert np.allclose(ref_fj, fj)

    def test_structure_factors(self):
        """ Check that accelerated structure factors calculation is correct. """
        U = np.random.randn(self.xyz.shape[0])
        sf = scatter.structure_factors(self.q_grid, self.xyz, self.ff_a, self.ff_b, self.ff_c, U)
        sf_ref = reference.structure_factors(self.q_grid, self.xyz, self.elements, U)
        assert np.allclose(np.square(np.abs(sf)), np.square(np.abs(sf_ref)))
        
class TestReference(object):
    """
    Check reference implementation of the diffuse calculation from a covariance matrix.
    """
    @classmethod
    def setup_class(cls):
        qmesh = np.mgrid[-2:2:21j,-2:2:21j,-2:2:21j]
        cls.map_shape = qmesh.shape[1:]
        cls.q_grid = qmesh.T.reshape(-1, 3)
        cls.xyz, cls.elements = extract_model_info("pentagon.pdb")
        
    def test_no_disorder(self):
        """ Check Bragg and diffuse components without disorder. """
        Viso = np.zeros((self.xyz.shape[0], self.xyz.shape[0]))
        Vaniso = np.zeros((self.xyz.shape[0], self.xyz.shape[0], 3, 3))
        Id_iso, Ib_iso = reference.diffuse_covmat(self.q_grid, self.xyz, self.elements, Viso)
        Id_aniso, Ib_aniso = reference.diffuse_covmat(self.q_grid, self.xyz, self.elements, Vaniso)
        Ib_ref = np.square(np.abs(reference.structure_factors(self.q_grid, self.xyz, self.elements, np.zeros(self.xyz.shape[0]))))
        # bragg components should match
        assert np.allclose(Ib_ref, Ib_iso)
        assert np.allclose(Ib_ref, Ib_aniso)
        # diffuse component should be zero
        assert np.allclose(0, Id_iso)
        assert np.allclose(0, Id_aniso)

    def test_uncorrelated_disorder(self):
        """ Check case of uncorrelated disorder. """
        adps = np.random.randn(self.xyz.shape[0]) # (n_atoms)
        adps_iso = np.diagflat(adps) # (n_atoms, n_atoms)
        adps_aniso = adps_iso[:,:,None,None] * np.eye(3)[None,None,:] # (n_atoms, n_atoms, 3, 3)
        Ib_ref = np.square(np.abs(reference.structure_factors(self.q_grid, self.xyz, self.elements, adps)))
        Id_iso, Ib_iso = reference.diffuse_covmat(self.q_grid, self.xyz, self.elements, adps_iso)
        Id_aniso, Ib_aniso = reference.diffuse_covmat(self.q_grid, self.xyz, self.elements, adps_aniso)
        # bragg components match
        assert np.allclose(Ib_ref, Ib_iso)
        assert np.allclose(Ib_ref, Ib_aniso)
        # diffuse components match and are radially symmetric
        assert np.allclose(Id_iso, Id_aniso)
        assert np.allclose(Id_iso.reshape(self.map_shape)[10,:,:], Id_iso.reshape(self.map_shape)[:,10,:])
        assert np.allclose(Id_aniso.reshape(self.map_shape)[:,:,10], Id_aniso.reshape(self.map_shape)[:,10,:])
        
