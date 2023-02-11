import numpy as np
import gemmi
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.models import *
from eryx.pdb import AtomicModel
from eryx.map_utils import generate_grid
from eryx.base import compute_molecular_transform
from eryx.base import compute_crystal_transform
from base import setup_model

class TestTransforms:
    """
    Check crystal and molecular transform calculations.
    """
    def setup_class(cls):
        cls.pdb_path = "pdbs/5zck.pdb"
        cls.pdb_path_p1 = "pdbs/5zck_p1.pdb"
        cls.hsampling = (-4,4,1)
        cls.ksampling = (-17,17,1)
        cls.lsampling = (-29,29,1)

    def test_crystal_transform(self):
        """ Check crystal transform calculation. """
        q_grid, I1 = compute_crystal_transform(self.pdb_path,
                                               self.hsampling,
                                               self.ksampling,
                                               self.lsampling,
                                               expand_p1=True,
                                               parallelize=None)
        q_grid, I2 = compute_crystal_transform(self.pdb_path_p1,
                                               self.hsampling,
                                               self.ksampling,
                                               self.lsampling,
                                               expand_p1=False,
                                               parallelize=None)
        # crystal transform is more sensitive to limited precision of pdb xyz
        assert np.allclose(np.corrcoef(I1.flatten(), I2.flatten())[0,1], 1)
        assert np.allclose(np.mean(np.abs(I1 - I2) / I2), 0, atol=0.01)

def molecular_transform_reference(pdb_path, model, hsampling, ksampling, lsampling):
    """ Reference implementation (no fancy symmetrization). """
    model = AtomicModel(pdb_path, expand_p1=True, frame=-1)
    q_grid, map_shape = generate_grid(model.A_inv, hsampling, ksampling, lsampling)
    
    I = np.zeros(q_grid.shape[0])
    for asu in range(model.xyz.shape[0]):
        I += np.square(np.abs(structure_factors(q_grid,
                                                model.xyz[asu],
                                                model.ff_a[asu], 
                                                model.ff_b[asu], 
                                                model.ff_c[asu],
                                                parallelize=None)))
    return I.reshape(map_shape)
        
class TestMolecularTransform:
    """ Check that molecular transform calculation is working. """
    
    def setup_class(cls):
        cls.Iref = {}
        for case in ['orthorhombic', 'trigonal', 'triclinic', 'tetragonal']:
            pdb_path, model, hsampling, ksampling, lsampling = setup_model(case, expand_p1=True)
            cls.Iref[case] = molecular_transform_reference(pdb_path, model, hsampling, ksampling, lsampling)
        cls.dmin = dict(zip(['orthorhombic', 'trigonal', 'triclinic', 'tetragonal'], [1.5,1.5,1.5,6.0]))

    def test_basic_usage(self):
        """ Check that two symmetrization approaches match reference. """
        for case in ['orthorhombic', 'trigonal', 'triclinic', 'tetragonal']:
            pdb_path, model, hsampling, ksampling, lsampling = setup_model(case, expand_p1=True)
            hkl, I1 = compute_molecular_transform(pdb_path, hsampling, ksampling, lsampling, expand_friedel=True, parallelize=None)
            hkl, I2 = compute_molecular_transform(pdb_path, hsampling, ksampling, lsampling, expand_friedel=False, parallelize=None)
            assert np.max(np.abs(self.Iref[case] -  I1)/I1) < 1e-2
            assert np.max(np.abs(self.Iref[case] -  I2)/I2) < 1e-2
            
    def test_masking(self):
        """ Check case of applying a mask to limit resolution. """
        for case in ['orthorhombic', 'trigonal', 'triclinic', 'tetragonal']:
            pdb_path, model, hsampling, ksampling, lsampling = setup_model(case, expand_p1=True)
            hkl, I1 = compute_molecular_transform(pdb_path, hsampling, ksampling, lsampling, expand_friedel=True, res_limit=self.dmin[case], parallelize=None)
            hkl, I2 = compute_molecular_transform(pdb_path, hsampling, ksampling, lsampling, expand_friedel=False, res_limit=self.dmin[case], parallelize=None)
            assert np.max(np.abs(self.Iref[case][I1!=0] -  I1[I1!=0])/I1[I1!=0]) < 1e-2
            assert np.max(np.abs(self.Iref[case][I2!=0] -  I2[I2!=0])/I2[I2!=0]) < 1e-2
    
    def test_expansion(self):
        """ Check that the expand_friedel=True option correctly expands to full reciprocal grid. """
        case = 'tetragonal'
        pdb_path, model, hsampling, ksampling, lsampling = setup_model(case, expand_p1=True)
        hkl, I1 = compute_molecular_transform(pdb_path, (0,10,1), ksampling, lsampling, expand_friedel=True, res_limit=self.dmin[case], parallelize=None)
        assert np.max(np.abs(self.Iref[case][I1!=0] -  I1[I1!=0])/I1[I1!=0]) < 1e-2
        
        case = 'triclinic'
        pdb_path, model, hsampling, ksampling, lsampling = setup_model(case, expand_p1=True)
        hkl, I1 = compute_molecular_transform(pdb_path, hsampling, ksampling, (0,15,2), expand_friedel=True, res_limit=self.dmin[case], parallelize=None)
        assert np.max(np.abs(self.Iref[case][I1!=0] -  I1[I1!=0])/I1[I1!=0]) < 1e-2
        
class TestRigidBodyTranslations:
    """
    Check translational disorder model.
    """
    def setup_class(cls):
        pdb_path = "pdbs/5zck.pdb"
        cls.model = RigidBodyTranslations(pdb_path, (-4,4,1), (-17,17,1), (-29,29,1), parallelize=None)
        
    def test_anisotropic_sigma(self):
        """ Check that maps with the same (an)isotropic sigma match. """
        sigma = np.random.uniform()
        isotropic = self.model.apply_disorder(sigma)
        anisotropic = self.model.apply_disorder(np.array([sigma*np.ones(3)]))
        assert np.allclose(isotropic, anisotropic)
        
    def test_optimize(self):
        """ Check that optimization identifies the right sigma. """
        sigma = np.random.uniform()
        target = self.model.transform.flatten() * (1 - np.exp(-1 * np.square(sigma)*np.square(self.model.q_mags)))
        target = target.reshape(self.model.map_shape)
        ccs, sigmas = self.model.optimize(np.random.uniform()*target, 0.01, 0.99, 30)
        assert np.argmax(ccs) == np.argmin(np.abs((sigmas - sigma)))

class TestLiquidLikeMotions:
    """
    Check liquid like motions model.
    """
    def setup_class(cls):
        cls.pdb_path = "pdbs/5zck.pdb"
        cls.model = LiquidLikeMotions(cls.pdb_path, (-5,5,2), (-13,13,2), (-20,20,2), expand_p1=True)

    def test_mask(self):
        """ Check that mask is only applied to out-of-bounds q-vectors. """
        q_grid, map_shape = generate_grid(AtomicModel(self.pdb_path).A_inv,
                                          (self.model.hsampling[0], self.model.hsampling[1], self.model.hsampling[2]), 
                                          (self.model.ksampling[0], self.model.ksampling[1], self.model.ksampling[2]), 
                                          (self.model.lsampling[0], self.model.lsampling[1], self.model.lsampling[2])) 
        mask = self.model.mask.copy().flatten()
        q_grid_sel = self.model.q_grid[np.where(mask==1)[0][:,np.newaxis],:]
        q_grid_sel = q_grid_sel.reshape(-1, q_grid_sel.shape[-1])
        assert np.allclose(q_grid, q_grid_sel)
        
    def test_optimize(self):
        """ Check that correct sigma and gamma are identified. """
        s = float(np.random.choice(np.linspace(0.05, 0.45, 6)))
        g = float(np.random.choice(np.linspace(0.5, 3.5, 6)))
        target = self.model.apply_disorder(s,g)[0]
        target = target[self.model.mask.flatten()==1].reshape(self.model.map_shape_nopad)
        ccs, sigmas, gammas = self.model.optimize(target, 0.05, 0.45, 0.5, 3.5, ns_search=6, ng_search=6)
        assert gammas[np.argmin(np.abs(gammas - g))] == self.model.opt_gamma
        assert sigmas[np.argmin(np.abs(sigmas - s))] == self.model.opt_sigma

class TestRigidBodyRotations:
    """
    Check the rotational disorder model.
    """
    def setup_class(cls):
        pdb_path = "pdbs/2ol9.pdb"
        cls.model = RigidBodyRotations(pdb_path, (-14,14,1), (-5,5,1), (-15,15,1))

    def test_optimize(self):
        """ Check that optimization identifies the correct sigma. """
        param = float(np.random.choice(np.linspace(1.0,5.0,3)))
        target = self.model.apply_disorder(param)[0].reshape(self.model.map_shape)
        self.model.optimize(target, 1.0, 5.0, n_search=3)
        assert self.model.opt_sigma == param
