import numpy as np
import gemmi
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.models import *
from eryx.pdb import AtomicModel
from eryx.map_utils import generate_grid

def test_compute_transform():
    """
    Check that crystal transform calculation from expanded asus and p1 match.
    """
    sampling = (-13,13,1)
    q_grid, I_from_p1 = compute_transform('crystal', "histidine_p1.pdb", sampling, sampling, sampling, expand_p1=False)
    q_grid, I_from_p212121 = compute_transform('crystal', "histidine.pdb", sampling, sampling, sampling, expand_p1=True)
    assert np.allclose(I_from_p1, I_from_p212121)
    
class TestTranslationalDisorder:
    """
    Check translational disorder model.
    """
    def setup_class(cls):
        pdb_path = "histidine.pdb"
        cls.model = TranslationalDisorder(pdb_path, (-5,5,2), (-10,10,2), (-10,10,2))
        
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
        cls.pdb_path = "histidine.pdb"
        cls.model = LiquidLikeMotions(cls.pdb_path, (-13,13,2), (-13,13,2), (-13,13,2))

    def test_dilate(self):
        """ Check that map was correctly dilated: q_mags should match. """
        q_mags_int = np.linalg.norm(self.model.q_grid_int, axis=1).reshape(self.model.map_shape_int)
        q_mags_int = self.model._dilate(q_mags_int, (self.model.hsampling[2], self.model.ksampling[2], self.model.lsampling[2]))
        q_mags_frac = self.model.q_mags.copy().reshape(self.model.map_shape)
        q_mags_frac[q_mags_int==0] = 0
        assert np.allclose(q_mags_frac, q_mags_int)

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