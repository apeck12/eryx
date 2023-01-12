import numpy as np
import scipy.signal
import scipy.spatial
from .pdb import AtomicModel
from .map_utils import *
from .scatter import structure_factors
from .stats import compute_cc

def compute_crystal_transform(pdb_path, hsampling, ksampling, lsampling, U=None, batch_size=10000, expand_p1=True):
    """
    Compute the crystal transform as the coherent sum of the
    asymmetric units. If expand_p1 is False, it is assumed 
    that the pdb contains asymmetric units as separate frames.
    
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

def compute_molecular_transform(pdb_path, hsampling, ksampling, lsampling, U=None, batch_size=10000, expand_p1=True, symmetrize='reciprocal'):
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
    symmetrize : str
        symmetrization mode, either real or reciprocal
        
    Returns
    -------
    q_grid : numpy.ndarray, (n_points, 3)
        q-vectors corresponding to flattened intensity map
    I : numpy.ndarray, 3d
        intensity map of the molecular transform
    """
    if symmetrize == 'real':
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
    elif symmetrize == 'reciprocal':
        model = AtomicModel(pdb_path, expand_p1=False, frame=0)
        q_grid, map_shape = generate_grid(model.A_inv, hsampling, ksampling, lsampling)
        I = incoherent_from_reciprocal(model, hsampling, ksampling, lsampling, U=U, batch_size=batch_size)

    else: 
        raise ValueError("Symmetrize must be real or reciprocal")
        
    return q_grid, I.reshape(map_shape)

def incoherent_from_reciprocal(model, hsampling, ksampling, lsampling, U=None, batch_size=10000):
    """
    Compute intensities as the incoherent sum of the scattering
    from asymmetric units by symmetrizing in reciprocal space.
    This reduces the number of q-vectors to evaluate by up to the
    number of asymmetric units. The strategy is to determine each
    reflection's set of symmetry-equivalents, map these 3d vectors
    to 1d space by raveling, and then computing the intensity only
    for hkl grid points that haven't already been processed. 
    
    Parameters
    ----------
    model : AtomicModel 
        instance of AtomicModel class
    hsampling : tuple, shape (3,)
        (min, max, interval) along h axis
    ksampling : tuple, shape (3,)
        (min, max, interval) along k axis
    lsampling : tuple, shape (3,)
        (min, max, interval) along l axis        
    U : numpy.ndarray, shape (n_atoms,)
        isotropic displacement parameters, applied to each asymmetric unit
    batch_size : int
        number of q-vectors to evaluate per batch

    Returns
    -------
    I : numpy.ndarray, 3d
        symmetrized intensity map
    """
    hkl_grid, map_shape = generate_grid(model.A_inv, hsampling, ksampling, lsampling, return_hkl=True)
    hkl_grid_sym = get_symmetry_equivalents(hkl_grid, model.sym_ops)
    ravel, map_shape_ravel = get_ravel_indices(hkl_grid_sym, (hsampling[2], ksampling[2], lsampling[2]))
    
    I_sym = np.zeros(ravel.shape)
    for asu in range(I_sym.shape[0]):
        q_asu = 2*np.pi*np.inner(model.A_inv.T, hkl_grid_sym[asu]).T
        if asu == 0:
            I_sym[asu] = np.square(np.abs(structure_factors(q_asu,
                                                            model.xyz[0],
                                                            model.ff_a[0],
                                                            model.ff_b[0],
                                                            model.ff_c[0],
                                                            U=U,
                                                            batch_size=batch_size)))
        else:
            intersect1d, comm1, comm2 = np.intersect1d(ravel[0], ravel[asu], return_indices=True)
            I_sym[asu][comm2] = I_sym[0][comm1]
            comm3 = np.arange(len(ravel[asu]))[~np.in1d(ravel[asu],ravel[0])]
            I_sym[asu][comm3] = np.square(np.abs(structure_factors(q_asu[comm3],
                                                                   model.xyz[0],
                                                                   model.ff_a[0],
                                                                   model.ff_b[0],
                                                                   model.ff_c[0],
                                                                   U=U,
                                                                   batch_size=batch_size)))
    I = np.sum(I_sym, axis=0).reshape(map_shape)
    return I

class TranslationalDisorder:
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, batch_size=10000, expand_p1=True):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1, batch_size)
        
    def _setup(self, pdb_path, expand_p1=True, batch_size=10000):
        """
        Set up class, including computing the molecular transform.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1 : bool
            if True, expand to p1 (i.e. if PDB corresponds to the asymmetric unit)
        batch_size : int
            number of q-vectors to evaluate per batch
        """
        self.q_grid, self.transform = compute_molecular_transform(pdb_path, 
                                                                  self.hsampling, 
                                                                  self.ksampling, 
                                                                  self.lsampling,
                                                                  batch_size=batch_size,
                                                                  expand_p1=expand_p1)
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
        ccs = compute_cc(Id, np.expand_dims(target.flatten(), axis=0))
        opt_index = np.argmax(ccs)
        self.opt_sigma = sigmas[opt_index]
        self.opt_map = Id[opt_index].reshape(self.map_shape)

        print(f"Optimal sigma: {self.opt_sigma}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas

class LiquidLikeMotions:
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, batch_size=10000, border=1, expand_p1=True):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, batch_size, border, expand_p1)
                
    def _setup(self, pdb_path, batch_size, border, expand_p1):
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
        expand_p1 : bool
            if True, pdb corresponds to asymmetric unit; expand to unit cell
        """
        # compute crystal transform at integral Miller indices and dilate
        self.q_grid_int, transform = compute_crystal_transform(pdb_path, 
                                                               (self.hsampling[0]-border, self.hsampling[1]+border, 1), 
                                                               (self.ksampling[0]-border, self.ksampling[1]+border, 1), 
                                                               (self.lsampling[0]-border, self.lsampling[1]+border, 1),
                                                               batch_size=batch_size,
                                                               expand_p1=expand_p1)
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
        ccs = compute_cc(Id, np.expand_dims(target.flatten(), axis=0))
        opt_index = np.argmax(ccs)
        self.opt_map = Id[opt_index].reshape(self.map_shape_nopad)
        
        sigmas = np.repeat(sigmas, len(gammas), axis=0)
        gammas = np.tile(gammas, len(sigmas))
        self.opt_sigma = sigmas[opt_index]
        self.opt_gamma = gammas[opt_index]

        print(f"Optimal sigma: {self.opt_sigma}, optimal gamma: {self.opt_gamma}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas, gammas

class RotationalDisorder:
    
    """
    Model of rigid body rotational disorder, in which all atoms in 
    each asymmetric unit rotate as a rigid unit around a randomly 
    oriented axis with a normally distributed rotation angle.
    """
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, batch_size=10000, expand_p1=True):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1)
        self.batch_size = batch_size
        
    def _setup(self, pdb_path, expand_p1):
        """
        Compute q-vectors to evaluate.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1 : bool
            if True, expand to p1 (i.e. if PDB corresponds to the asymmetric unit)
        """
        self.model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
        self.q_grid, self.map_shape = generate_grid(self.model.A_inv, 
                                                    self.hsampling, 
                                                    self.ksampling, 
                                                    self.lsampling)
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
    
    @staticmethod
    def axis_angle_to_quaternion(axis, theta):
        """
        Convert an angular rotation around an axis series to quaternions.

        Parameters
        ----------
        axis : numpy.ndarray, size (num_pts, 3)
            axis vector defining rotation
        theta : numpy.ndarray, size (num_pts)
            angle in radians defining anticlockwise rotation around axis

        Returns
        -------
        quat : numpy.ndarray, size (num_pts, 4)
            quaternions corresponding to axis/theta rotations
        """
        axis /= np.linalg.norm(axis, axis=1)[:,None]
        angle = theta / 2

        quat = np.zeros((len(theta), 4))
        quat[:,0] = np.cos(angle)
        quat[:,1:] = np.sin(angle)[:,None] * axis

        return quat

    @staticmethod
    def generate_rotations_around_axis(sigma, num_rot, axis=np.array([0,0,1.0])):
        """
        Generate uniform random rotations about an axis.

        Parameters
        ----------
        sigma : float
            standard deviation of angular sampling around axis in degrees
        num_rot : int
            number of rotations to generate
        axis : numpy.ndarray, shape (3,)
            axis about which to generate rotations

        Returns
        -------
        rot_mat : numpy.ndarray, shape (num, 3, 3)
            rotation matrices
        """
        axis /= np.linalg.norm(axis)
        random_R = scipy.spatial.transform.Rotation.random(num_rot).as_matrix()
        random_ax = np.inner(random_R, np.array([0,0,1.0]))
        thetas = np.deg2rad(sigma) * np.random.randn(num_rot)
        rot_vec = thetas[:,np.newaxis] * random_ax
        rot_mat = scipy.spatial.transform.Rotation.from_rotvec(rot_vec).as_matrix()
        return rot_mat
    
    def apply_disorder(self, sigmas, num_rot=100):
        """
        Compute the diffuse maps(s) resulting from rotational disorder for 
        the given sigmas by applying Guinier's equation to an ensemble of 
        rotated molecules, and then taking the incoherent sum of all of the
        asymmetric units.
        
        Parameters
        ----------
        sigmas : float or array of shape (n_sigma,) 
            standard deviation(s) of angular sampling in degrees
        num_rot : int
            number of rotations to generate per sigma
        
        Returns
        -------
        Id : numpy.ndarray, (n_sigma, q_grid.shape[0])
            diffuse intensity maps for the corresponding parameters
        """
        if type(sigmas) == float or type(sigmas) == int:
            sigmas = np.array([sigmas])
            
        Id = np.zeros((len(sigmas), self.q_grid.shape[0]))
        for n_sigma,sigma in enumerate(sigmas):
            for asu in range(self.model.n_asu):
                # rotate atomic coordinates
                rot_mat = self.generate_rotations_around_axis(sigma, num_rot)
                com = np.mean(self.model.xyz[asu], axis=0)
                xyz_rot = np.matmul(self.model.xyz[asu] - com, rot_mat) 
                xyz_rot += com
                
                # apply Guinier's equation to rotated ensemble
                fc = np.zeros(self.q_grid.shape[0], dtype=complex)
                fc_square = np.zeros(self.q_grid.shape[0])
                for rnum in range(num_rot):
                    A = structure_factors(self.q_grid, 
                                          xyz_rot[rnum], 
                                          self.model.ff_a[asu], 
                                          self.model.ff_b[asu], 
                                          self.model.ff_c[asu], 
                                          U=None, 
                                          batch_size=10000)
                    fc += A
                    fc_square += np.square(np.abs(A)) 
                Id[n_sigma] += fc_square / num_rot - np.square(np.abs(fc / num_rot))

        return Id 
    
    def optimize(self, target, sigma_min, sigma_max, n_search=20, num_rot=100):
        """
        Scan to find the sigma that maximizes the overall Pearson
        correlation between the target and computed maps. 
        
        Parameters
        ----------
        target : numpy.ndarray, 3d
            target map, of shape self.map_shape
        sigma_min : float 
            lower bound of sigma
        sigma_max : float 
            upper bound of sigma
        n_search : int
            sampling frequency between sigma_min and sigma_max
        num_rot : int
            number of rotations to generate per sigma
        
        Returns
        -------
        ccs : numpy.ndarray, shape (n_search,)
            Pearson correlation coefficients to target maps
        sigmas : numpy.ndarray, shape (n_search,) or (n_search, n_search, n_search)
            sigmas that were scanned over, ordered as ccs
        """
        assert target.shape == self.map_shape
        
        sigmas = np.linspace(sigma_min, sigma_max, n_search)        
        Id = self.apply_disorder(sigmas, num_rot)
        ccs = compute_cc(Id, np.expand_dims(target.flatten(), axis=0))
        opt_index = np.argmax(ccs)
        self.opt_sigma = sigmas[opt_index]
        self.opt_map = Id[opt_index].reshape(self.map_shape)

        print(f"Optimal sigma: {self.opt_sigma}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas

class EnsembleDisorder:
    
    """
    Model of ensemble disorder, in which the components of the
    asymmetric unit populate distinct biological states.
    """
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, batch_size=10000, expand_p1=True):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1)
        self.batch_size = batch_size
        
    def _setup(self, pdb_path, expand_p1):
        """
        Compute q-vectors to evaluate.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1 : bool
            if True, expand to p1 (i.e. if PDB corresponds to the asymmetric unit)
        """
        self.model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
        self.q_grid, self.map_shape = generate_grid(self.model.A_inv, 
                                                    self.hsampling, 
                                                    self.ksampling, 
                                                    self.lsampling)
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        
    def apply_disorder(self, weights=None):
        """
        Compute the diffuse maps(s) resulting from ensemble disorder using
        Guinier's equation, and then taking the incoherent sum of all the
        asymmetric units. 
        
        Parameters
        ----------
        weights : shape (n_sets, n_conf) 
            set(s) of probabilities associated with each conformation

        Returns
        -------
        Id : numpy.ndarray, (n_sets, q_grid.shape[0])
            diffuse intensity map for the corresponding parameters
        """
        if weights is None:
            weights = 1.0 / self.model.n_conf * np.array([np.ones(self.model.n_conf)])
        if len(weights.shape) == 1:
            weights = np.array([weights])
        if weights.shape[1] != self.model.n_conf:
            raise ValueError("Second dimension of weights must match number of conformations.")
            
        n_maps = weights.shape[0]
        Id = np.zeros((weights.shape[0], self.q_grid.shape[0]))

        for asu in range(self.model.n_asu):

            fc = np.zeros((weights.shape[0], self.q_grid.shape[0]), dtype=complex)
            fc_square = np.zeros((weights.shape[0], self.q_grid.shape[0]))

            for conf in range(self.model.n_conf):
                index = conf * self.model.n_asu + asu
                A = structure_factors(self.q_grid, 
                                      self.model.xyz[index], 
                                      self.model.ff_a[index], 
                                      self.model.ff_b[index], 
                                      self.model.ff_c[index], 
                                      U=None, 
                                      batch_size=10000)
                for nm in range(n_maps):
                    fc[nm] += A * weights[nm][conf]
                    fc_square[nm] += np.square(np.abs(A)) * weights[nm][conf]

            for nm in range(n_maps):
                Id[nm] += fc_square[nm] - np.square(np.abs(fc[nm]))
                
        return Id
