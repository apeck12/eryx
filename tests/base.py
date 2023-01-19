import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.pdb import AtomicModel

def setup_model(case, expand_p1=False):
    """ Return variables for a test case in the given space group. """
    
    if case == 'orthorhombic':
        hsampling = (-5,5,3)
        ksampling = (-13,13,2)
        lsampling = (-20,20,2)
        pdb_path = "pdbs/5zck.pdb"

    elif case == 'trigonal':
        hsampling = (-18,18,2)
        ksampling = (-18,18,2)
        lsampling = (-5,5,3)
        pdb_path = "pdbs/7n2h.pdb"

    elif case == 'triclinic':
        hsampling = (-14, 14, 2)
        ksampling = (-5, 5, 2)
        lsampling = (-15, 15, 2)
        pdb_path = "pdbs/2ol9.pdb"
        
    elif case == 'tetragonal':
        hsampling = (-10, 10, 1)
        ksampling = (-10, 10, 1)
        lsampling = (-6, 6, 2)
        pdb_path = "pdbs/193l.pdb"   
        
    model = AtomicModel(pdb_path, expand_p1=expand_p1)
    return pdb_path, model, hsampling, ksampling, lsampling
