import numpy as np
import scipy.signal
from .pdb import AtomicModel
from .map_utils import generate_grid, pearson_cc
from .scatter import structure_factors

def compute_crystal_transform(pdb_path, hsampling, ksampling, lsampling, U=None, batch_size=10000, expand_p1=True):
    """
    Compute the molecular transform as the incoherent sum
    of the asymmetric units. If expand_p1 is False, it is
    assumed that the pdb contains the asymmetric units as
    separate frames / models.
    
    Parameters
    ----------
    pdb_path : str
        path to coordinates file 
    hsampling : tuple, shape (3,)
        (hmin, hmax, oversampling) relative to Miller indices
    ksampling : tuple, shape (3,)
        (kmin, kmax, oversampling) relative to Miller indices
    lsampling : tuple, shape (3,)
        (lmin, lmax, oversampling) relative to Miller indices
    U : numpy.ndarray, shape (n_atoms,)
        isotropic displacement parameters, applied to each asymmetric unit
    batch_size : int
        number of q-vectors to evaluate per batch
    expand_p1 : bool
        if True, expand PDB (asymmetric unit) to unit cell
        
    Returns
    -------
    q_grid : numpy.ndarray, (n_points, 3)
        q-vectors corresponding to flattened intensity map
    I : numpy.ndarray, 3d
        intensity map of the molecular transform
    """
    model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
    model.flatten_model()
    q_grid, map_shape = generate_grid(model.A_inv, hsampling, ksampling, lsampling)
    
    I = np.square(np.abs(structure_factors(q_grid,
                                           model.xyz, 
                                           model.ff_a,
                                           model.ff_b,
                                           model.ff_c,
                                           U=U, 
                                           batch_size=batch_size)))
    return q_grid, I.reshape(map_shape)

def compute_molecular_transform(pdb_path, hsampling, ksampling, lsampling, U=None, batch_size=10000, expand_p1=True):
    """
    Compute the molecular transform as the incoherent sum
    of the asymmetric units. If expand_p1 is False, it is
    assumed that the pdb contains the asymmetric units as
    separate frames / models.
    
    Parameters
    ----------
    pdb_path : str
        path to coordinates file 
    hsampling : tuple, shape (3,)
        (hmin, hmax, oversampling) relative to Miller indices
    ksampling : tuple, shape (3,)
        (kmin, kmax, oversampling) relative to Miller indices
    lsampling : tuple, shape (3,)
        (lmin, lmax, oversampling) relative to Miller indices
    U : numpy.ndarray, shape (n_atoms,)
        isotropic displacement parameters, applied to each asymmetric unit
    batch_size : int
        number of q-vectors to evaluate per batch
    expand_p1 : bool
        if True, expand PDB (asymmetric unit) to unit cell
        
    Returns
    -------
    q_grid : numpy.ndarray, (n_points, 3)
        q-vectors corresponding to flattened intensity map
    I : numpy.ndarray, 3d
        intensity map of the molecular transform
    """
    model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
    q_grid, map_shape = generate_grid(model.A_inv, hsampling, ksampling, lsampling)
    
    I = np.zeros(q_grid.shape[0])
    for asu in range(model.xyz.shape[0]):
        I += np.square(np.abs(structure_factors(q_grid, 
                                                model.xyz[asu], 
                                                model.ff_a[asu], 
                                                model.ff_b[asu], 
                                                model.ff_c[asu], 
                                                U=U, 
                                                batch_size=batch_size)))
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
    
    def optimize(self, target, sigmas_min, sigmas_max, n_search=20):
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
        self.opt_sigma = sigmas[opt_index]
        self.opt_map = Id[opt_index].reshape(self.map_shape)

        print(f"Optimal sigma: {self.opt_sigma}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas

class LiquidLikeMotions:
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, batch_size=10000, border=1):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, batch_size, border)
                
    def _setup(self, pdb_path, batch_size, border):
        """
        Set up class, including calculation of the crystal transform.
        The transform can be evaluated to a higher resolution so that
        the edge of the disorder map doesn't encounter a boundary.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        batch_size : int
            number of q-vectors to evaluate per batch
        border : int
            number of border Miller indices along each direction 
        """
        # compute crystal transform at integral Miller indices and dilate
        self.q_grid_int, transform = compute_transform('crystal', 
                                                       pdb_path, 
                                                       (self.hsampling[0]-border, self.hsampling[1]+border, 1), 
                                                       (self.ksampling[0]-border, self.ksampling[1]+border, 1), 
                                                       (self.lsampling[0]-border, self.lsampling[1]+border, 1),
                                                       batch_size=batch_size)
        self.transform = self._dilate(transform, (self.hsampling[2], self.ksampling[2], self.lsampling[2]))
        self.map_shape, self.map_shape_int = self.transform.shape, transform.shape
        
        # generate q-vectors for dilated map
        model = AtomicModel(pdb_path)
        self.q_grid, self.map_shape = generate_grid(model.A_inv, 
                                                    (self.hsampling[0]-border, self.hsampling[1]+border, self.hsampling[2]), 
                                                    (self.ksampling[0]-border, self.ksampling[1]+border, self.ksampling[2]), 
                                                    (self.lsampling[0]-border, self.lsampling[1]+border, self.lsampling[2])) 
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        
        # generate mask for padded region
        self.mask = np.zeros(self.map_shape)
        self.mask[border*self.hsampling[2]:-border*self.hsampling[2],
                  border*self.ksampling[2]:-border*self.ksampling[2],
                  border*self.lsampling[2]:-border*self.lsampling[2]] = 1
        self.map_shape_nopad = tuple(np.array(self.map_shape) - np.array([2*border*self.hsampling[2], 
                                                                          2*border*self.ksampling[2], 
                                                                          2*border*self.lsampling[2]]))
        
    def _dilate(self, X, d):
        """
        Dilate map, placing zeros between the original entries.
        
        Parameters
        ----------
        X : numpy.ndarray, 3d
            map to dilate
        d : tuple, shape (3,)
            number of zeros between entries along each direction
            
        Returns
        -------
        Xd : numpy.ndarray, 3d
            dilated map
        """
        Xd_shape = np.multiply(X.shape, d)
        Xd = np.zeros(Xd_shape, dtype=X.dtype)
        Xd[0:Xd_shape[0]:d[0], 0:Xd_shape[1]:d[1], 0:Xd_shape[2]:d[2]] = X
        return Xd[:-1*(d[0]-1),:-1*(d[1]-1),:-1*(d[2]-1)]
    
    def apply_disorder(self, sigmas, gammas):
        """
        Compute the diffuse map(s) from the crystal transform as:
        I_diffuse = q2s2 * np.exp(-q2s2) * [I_transform * kernel(q)]
        where q2s2 = q^2 * s^2, and the kernel models covariances as 
        decaying exponentially with interatomic distance: 
        kernel(q) = 8 * pi * gamma^3 / (1 + q^2 * gamma^2)^2
        
        Parameters
        ----------
        sigmas : float or array of shape (n_sigma,) or (n_sigma, 3)
            (an)isotropic displacement parameter for asymmetric unit 
        gammas : float or array of shape (n_gamma,)
            kernel's correlation length
            
        Returns
        -------
        Id : numpy.ndarray, (n_sigma*n_gamma, q_grid.shape[0])
            diffuse intensity maps for the corresponding parameters
        """
        
        if type(gammas) == float or type(gammas) == int:
            gammas = np.array([gammas])   
        if type(sigmas) == float or type(sigmas) == int:
            sigmas = np.array([sigmas])

        # generate kernel and convolve with transform
        Id = np.zeros((len(gammas), self.q_grid.shape[0]))
        kernels = 8.0 * np.pi * (gammas[:,np.newaxis]**3) / np.square(1 + np.square(gammas[:,np.newaxis] * self.q_mags))
        for num in range(len(gammas)):
            Id[num] = scipy.signal.fftconvolve(self.transform, kernels[num].reshape(self.map_shape), mode='same').flatten()
        Id = np.tile(Id, (len(sigmas), 1))

        # scale with displacement parameters
        if len(sigmas.shape)==1:
            sigmas = np.repeat(sigmas, len(gammas))
            q2s2 = np.square(self.q_mags) * np.square(sigmas)[:,np.newaxis]
        else:
            sigmas = np.repeat(sigmas, len(gammas), axis=0)
            q2s2 = np.sum(self.q_grid.T * np.dot(np.square(sigmas)[:,np.newaxis] * np.eye(3), self.q_grid.T), axis=1)

        Id *= np.exp(-1*q2s2) * q2s2
        return Id

    def optimize(self, target, sigmas_min, sigmas_max, gammas_min, gammas_max, ns_search=20, ng_search=10):
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
        gammas_min : float 
            lower bound of gamma
        gammas_max : float 
            upper bound of gamma
        ns_search : int
            sampling frequency between sigmas_min and sigmas_max
        ng_search : int
            sampling frequency between gammas_min and gammas_max
        
        Returns
        -------
        ccs : numpy.ndarray, shape (n_search,)
            Pearson correlation coefficients to target maps
        sigmas : numpy.ndarray, shape (n_search,) or (n_search, n_search, n_search)
            sigmas that were scanned over, ordered as ccs
        """
        assert target.shape == self.map_shape_nopad
        
        if (type(sigmas_min) == float) and (type(sigmas_max) == float):
            sigmas = np.linspace(sigmas_min, sigmas_max, ns_search)
        else:
            sa, sb, sc = [np.linspace(sigmas_min[i], sigmas_max[i], ns_search) for i in range(3)]
            sigmas = np.array(list(itertools.product(sa, sb, sc)))
        gammas = np.linspace(gammas_min, gammas_max, ng_search)
        
        Id = self.apply_disorder(sigmas, gammas)
        Id = Id[:,self.mask.flatten()==1]
        ccs = pearson_cc(Id, np.expand_dims(target.flatten(), axis=0))
        opt_index = np.argmax(ccs)
        self.opt_map = Id[opt_index].reshape(self.map_shape_nopad)
        
        sigmas = np.repeat(sigmas, len(gammas), axis=0)
        gammas = np.tile(gammas, len(sigmas))
        self.opt_sigma = sigmas[opt_index]
        self.opt_gamma = gammas[opt_index]

        print(f"Optimal sigma: {self.opt_sigma}, optimal gamma: {self.opt_gamma}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas, gammas
