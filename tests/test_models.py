import numpy as np
import gemmi
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.models import *

class TestTranslationalDisorder:
    """
    Check translational disorder model
    """
    @classmethod
    def setup_class(cls):
        pdb_path = "histidine.pdb"
        cls.model = TranslationalDisorder(pdb_path, (-5,5,2), (-10,10,2), (-10,10,2))
        
    def test_anisotropic_sigma(self):
        """ Check that maps with the same (an)isotropic sigma match. """
        sigma = np.random.uniform()
        isotropic = self.model.apply_disorder(sigma)
        anisotropic = self.model.apply_disorder(np.array([sigma*np.ones(3)]))
        assert np.allclose(isotropic, anisotropic)
        
    def test_optimize_sigma(self):
        """ Check that optimization identifies the right sigma. """
        sigma = np.random.uniform()
        target = self.model.transform.flatten() * (1 - np.exp(-1 * sigma*np.square(self.model.q_mags)))
        target = target.reshape(self.model.map_shape)
        ccs, sigmas = self.model.optimize_sigma(np.random.uniform()*target, 0.01, 0.99, 30)
        assert np.argmax(ccs) == np.argmin(np.abs((sigmas - sigma)))
