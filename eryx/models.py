import numpy as np
from .pdb import AtomicModel
from .map_utils import generate_grid, pearson_cc
from .scatter import structure_factors

def compute_transform(transform, pdb_path, hsampling, ksampling, lsampling,
                      U=None, batch_size=10000, expand_p1=True):
    """
    Compute either the crystal transform or the molecular transform
    as the coherent and incoherent sums of the scattering from each 
    asymmetric unit, respectively.
    
    Parameters
    ----------
    transform : str
        molecular or crystal
    pdb_path : str
        path to coordinates file of asymmetric unit
    hsampling : tuple, shape (3,)
        (hmin, hmax, oversampling relative to Miller indices)
    ksampling : tuple, shape (3,)
        (kmin, kmax, oversampling relative to Miller indices)
    lsampling : tuple, shape (3,)
        (lmin, lmax, oversampling relative to Miller indices)
    U : numpy.ndarray, shape (n_atoms,)
        isotropic displacement parameters, applied to each asymmetric unit
    batch_size : int
        number of q-vectors to evaluate per batch
        
    Returns
    -------
    q_grid : numpy.ndarray, (n_points, 3)
        q-vectors corresponding to flattened intensity map
    I : numpy.ndarray, 3d
        intensity map of the molecular transform
    """
    if transform=='molecular':
        model = AtomicModel(pdb_path, expand_p1=expand_p1)
    elif transform=='crystal':
        model = AtomicModel(pdb_path, expand_p1=expand_p1)
        if expand_p1:
            model.concatenate_asus()
        model.xyz = np.expand_dims(model.xyz, axis=0)
    else:
        raise ValueError("Transform type not recognized. Must be molecular or crystal")
        
    q_grid, map_shape = generate_grid(model.A_inv, hsampling, ksampling, lsampling)
    I = np.zeros(q_grid.shape[0])
    for asu in range(model.xyz.shape[0]):
        A = structure_factors(q_grid, model.xyz[asu], model.ff_a, model.ff_b, model.ff_c, U=U, batch_size=batch_size)
        I += np.square(np.abs(A))
    return q_grid, I.reshape(map_shape)

class TranslationalDisorder:
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, batch_size=10000):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, batch_size)
        
    def _setup(self, pdb_path, batch_size):
        """
        Set up class, including computing the molecular transform.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        batch_size : int
            number of q-vectors to evaluate per batch
        """
        self.q_grid, self.transform = compute_transform('molecular', 
                                                        pdb_path, 
                                                        self.hsampling, 
                                                        self.ksampling, 
                                                        self.lsampling,
                                                        batch_size=batch_size)
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        self.map_shape = self.transform.shape
        self.map_optimized = None
    
    def apply_disorder(self, sigmas):
        """
        Compute the diffuse map(s) from the molecular transform:
        I_diffuse = I_transform * (1 - q^2 * sigma^2)
        for a single sigma or set of (an)isotropic sigmas.

        Parameters
        ----------
        sigma : float or array of shape (n_sigma,) or (n_sigma, 3)
            (an)isotropic displacement parameter for asymmetric unit 

        Returns
        -------
        Id : numpy.ndarray, (n_sigma, q_grid.shape[0])
            diffuse intensity maps for the corresponding sigma(s)
        """
        if type(sigmas) == float:
            sigmas = np.array([sigmas])

        if len(sigmas.shape) == 1:
            wilson = np.square(self.q_mags) * np.square(sigmas)[:,np.newaxis]
        else:
            wilson = np.sum(self.q_grid.T * np.dot(np.square(sigmas)[:,np.newaxis] * np.eye(3), self.q_grid.T), axis=1)

        Id = self.transform.flatten() * (1 - np.exp(-1 * wilson))
        return Id
    
    def optimize_sigma(self, target, sigmas_min, sigmas_max, n_search=20):
        """
        Scan to find the sigma that maximizes the overall Pearson
        correlation between the target and computed maps. 
        
        Parameters
        ----------
        target : numpy.ndarray, 3d
            target map, of shape self.map_shape
        sigmas_min : float or tuple of shape (3,)
            lower bound of (an)isotropic sigmas
        sigmas_max : float or tuple of shape (3,)
            upper bound of (an)isotropic sigmas, same type/dimension as sigmas_min
        n_search : int
            sampling frequency between sigmas_min and sigmas_max
        
        Returns
        -------
        ccs : numpy.ndarray, shape (n_search,)
            Pearson correlation coefficients to target maps
        sigmas : numpy.ndarray, shape (n_search,) or (n_search, n_search, n_search)
            sigmas that were scanned over, ordered as ccs
        """
        assert target.shape == self.map_shape
        
        if (type(sigmas_min) == float) and (type(sigmas_max) == float):
            sigmas = np.linspace(sigmas_min, sigmas_max, n_search)
        else:
            sa, sb, sc = [np.linspace(sigmas_min[i], sigmas_max[i], n_search) for i in range(3)]
            sigmas = np.array(list(itertools.product(sa, sb, sc)))
        
        Id = self.apply_disorder(sigmas)
        ccs = pearson_cc(Id, np.expand_dims(target.flatten(), axis=0))
        opt_index = np.argmax(ccs)
        self.map_optimized = Id[opt_index].reshape(self.map_shape)

        print(f"Optimal sigma: {sigmas[opt_index]}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas
