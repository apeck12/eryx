import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eryx.stats as stats

def test_compute_cc():
    """ Check that Pearson correlation coefficient is correctly calculated. """
    
    # check when all values of arrays are valid
    a = np.random.randn(18).reshape(3,6)
    b = np.array([np.random.randn(6)])
    est_cc = stats.compute_cc(a,b)
    ref_cc = [np.corrcoef(a[i], b[0])[0,1] for i in range(a.shape[0])]
    assert np.allclose(est_cc, ref_cc)
    
    # check when there's a nan value
    index = np.random.randint(0, high=a.shape[1])
    b[:,index] = np.nan
    est_cc = stats.compute_cc(a,b)
    ref_cc = [np.corrcoef(a[i][~np.isnan(b[0])], b[0][~np.isnan(b[0])])[0,1] for i in range(a.shape[0])]
    assert np.allclose(est_cc, ref_cc)
