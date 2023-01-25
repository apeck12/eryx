import numpy as np
import scipy.signal
import scipy.spatial
import glob
import os
from tqdm import tqdm
from .pdb import AtomicModel, GaussianNetworkModel
from .map_utils import *
from .scatter import structure_factors
from .stats import compute_cc
from .base import compute_molecular_transform, compute_crystal_transform

class RigidBodyTranslations:
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_friedel=True, res_limit=0, batch_size=10000, n_processes=8):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_friedel, res_limit, batch_size, n_processes)
        
    def _setup(self, pdb_path, expand_friedel=True, res_limit=0, batch_size=10000, n_processes=8):
        """
        Set up class, including computing the molecular transform.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_friedel : bool
            if True, expand to include portion of reciprocal space related by Friedel's law
        res_limit : float
            high resolution limit
        batch_size : int     
            number of q-vectors to evaluate per batch
        n_processes : int
            number of processors for structure factor calculation
        """
        self.q_grid, self.transform = compute_molecular_transform(pdb_path, 
                                                                  self.hsampling, 
                                                                  self.ksampling, 
                                                                  self.lsampling,
                                                                  expand_friedel=expand_friedel,
                                                                  res_limit=res_limit,
                                                                  batch_size=batch_size,
                                                                  n_processes=n_processes)
        self.transform[self.transform==0] = np.nan # compute_cc expects masked values to be np.nan
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
    
    """
    Model in which collective motions decay exponentially with distance
    across the crystal. Mathematically the predicted diffuse scattering 
    is the convolution between the crystal transform and a disorder kernel.
    In the asu_confined regime, disorder is confined to the asymmetric unit
    and the convolution is with the molecular rather than crystal transform.
    """
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_p1=True, 
                 border=1, res_limit=0, batch_size=5000, n_processes=8, asu_confined=False):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1, border, res_limit, batch_size, n_processes, asu_confined)
                
    def _setup(self, pdb_path, expand_p1, border, res_limit, batch_size, n_processes, asu_confined):
        """
        Set up class, including calculation of the crystal or molecular 
        transform for the classic and asu-confined variants of the LLM,
        respectively. The transform is evaluated to a higher resolution 
        to reduce convolution artifacts at the map's boundary.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1 : bool
            if True, pdb corresponds to asymmetric unit; expand to unit cell
        border : int
            number of border (integral) Miller indices along each direction 
        res_limit : float
            high-resolution limit in Angstrom
        batch_size : int
            number of q-vectors to evaluate per batch
        n_processes : int
            number of processes for structure factor calculation
        asu_confined : bool
            False for crystal transform, True for molecular trasnsform
        """
        # generate atomic model
        model = AtomicModel(pdb_path, expand_p1=expand_p1)
        model.flatten_model()
        
        # get grid for padded map
        hsampling_padded = (self.hsampling[0]-border, self.hsampling[1]+border, self.hsampling[2])
        ksampling_padded = (self.ksampling[0]-border, self.ksampling[1]+border, self.ksampling[2])
        lsampling_padded = (self.lsampling[0]-border, self.lsampling[1]+border, self.lsampling[2])
        hkl_grid, self.map_shape = generate_grid(model.A_inv, 
                                                 hsampling_padded,
                                                 ksampling_padded,
                                                 lsampling_padded,
                                                 return_hkl=True)
        self.res_mask, res_map = get_resolution_mask(model.cell, hkl_grid, res_limit)
        
        # compute crystal or molecular transform
        if not asu_confined:
            self.q_grid, self.transform = compute_crystal_transform(pdb_path,
                                                                    hsampling_padded,
                                                                    ksampling_padded,
                                                                    lsampling_padded,
                                                                    expand_p1=expand_p1,
                                                                    res_limit=res_limit,
                                                                    batch_size=batch_size,
                                                                    n_processes=n_processes)
        else:
            self.q_grid, self.transform = compute_molecular_transform(pdb_path,
                                                                      hsampling_padded,
                                                                      ksampling_padded,
                                                                      lsampling_padded,
                                                                      expand_p1=expand_p1,
                                                                      expand_friedel=False,
                                                                      res_limit=res_limit,
                                                                      batch_size=batch_size, 
                                                                      n_processes=n_processes)
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        
        # generate mask for padded region
        self.mask = np.zeros(self.map_shape)
        self.mask[border*self.hsampling[2]:-border*self.hsampling[2],
                  border*self.ksampling[2]:-border*self.ksampling[2],
                  border*self.lsampling[2]:-border*self.lsampling[2]] = 1
        self.map_shape_nopad = tuple(np.array(self.map_shape) - np.array([2*border*self.hsampling[2], 
                                                                          2*border*self.ksampling[2], 
                                                                          2*border*self.lsampling[2]]))

        # empty lists to populate with all sigmas/gammas that have been scanned
        self.scan_sigmas = []
        self.scan_gammas = []
        self.scan_ccs = []
        
    def fft_convolve(self, transform, kernel):
        """ 
        Convolve the transform and kernel by multiplying their 
        Fourier transforms. This approach requires less memory 
        than scipy.signal.fftconvolve.

        Parameters
        ----------
        transform : numpy.ndarray, 3d
            crystal or molecular transform map
        kernel : numpy.ndarray, 3d
            disorder kernel, same shape as transform

        Returns
        -------
        conv : numpy.ndarray 
            convolved map
        """
        ft_transform = np.fft.fftn(transform)
        ft_kernel = np.fft.fftn(kernel/kernel.sum()) 
        ft_conv = ft_transform * ft_kernel
        return np.fft.ifftshift(np.fft.ifftn(ft_conv).real)

    def plot_scan(self, output=None):
        """
        Plot a heatmap of the overall correlation coefficient
        as a function of sigma and gamma.
        Parameters
        ----------
        output : str
            if provided, save plot to given path
        """
        import matplotlib.pyplot as plt

        sigmas = np.array(self.scan_sigmas)
        gammas = np.array(self.scan_gammas)
        ccs = np.array(self.scan_ccs)

        xi = np.linspace(sigmas.min(), sigmas.max(), 25)
        yi = np.linspace(gammas.min(), gammas.max(), 25)
        zi = scipy.interpolate.griddata((sigmas, gammas), ccs, (xi[None,:], yi[:,None]), method='cubic')

        plt.contourf(xi,yi,zi,25,linewidths=0.5)
        plt.xlabel("$\sigma$ ($\mathrm{\AA}$)", fontsize=14)
        plt.ylabel("$\gamma$ ($\mathrm{\AA}$)", fontsize=14)
        cb = plt.colorbar()
        cb.ax.set_ylabel("CC", fontsize=14)

        if output is not None:
            plt.savefig(output, dpi=300, bbox_inches='tight')
    
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
            if np.prod(self.map_shape) < 1e7:
                Id[num] = scipy.signal.fftconvolve(self.transform,
                                                   kernels[num].reshape(self.map_shape)/np.sum(kernels[num]),
                                                   mode='same').flatten()
            else:
                Id[num] = self.fft_convolve(self.transform, kernels[num].reshape(self.map_shape)).flatten()
        Id = np.tile(Id, (len(sigmas), 1))

        # scale with displacement parameters
        if len(sigmas.shape)==1:
            sigmas = np.repeat(sigmas, len(gammas))
            q2s2 = np.square(self.q_mags) * np.square(sigmas)[:,np.newaxis]
        else:
            sigmas = np.repeat(sigmas, len(gammas), axis=0)
            q2s2 = np.sum(self.q_grid.T * np.dot(np.square(sigmas)[:,np.newaxis] * np.eye(3), self.q_grid.T), axis=1)

        Id *= np.exp(-1*q2s2) * q2s2
        Id[:,~self.res_mask] = np.nan
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
        if self.scan_ccs and np.max(ccs) > np.max(np.array(self.scan_ccs)):
            self.opt_map = Id[opt_index].reshape(self.map_shape_nopad)
        
        sigmas = np.repeat(sigmas, len(gammas), axis=0)
        gammas = np.tile(gammas, int(len(sigmas)/len(gammas)))
        self.opt_sigma = sigmas[opt_index]
        self.opt_gamma = gammas[opt_index]
        self.scan_sigmas.extend(list(sigmas))
        self.scan_gammas.extend(list(gammas))
        self.scan_ccs.extend(list(ccs))
        
        print(f"Optimal sigma: {self.opt_sigma}, optimal gamma: {self.opt_gamma}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas, gammas
    
class RigidBodyRotations:
    
    """
    Model of rigid body rotational disorder, in which all atoms in 
    each asymmetric unit rotate as a rigid unit around a randomly 
    oriented axis with a normally distributed rotation angle.
    """
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_p1=True, res_limit=0, batch_size=10000, n_processes=8):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1, res_limit)
        self.batch_size = batch_size
        self.n_processes = n_processes 
        
    def _setup(self, pdb_path, expand_p1, res_limit=0):
        """
        Compute q-vectors to evaluate.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1 : bool
            if True, expand to p1 (i.e. if PDB corresponds to the asymmetric unit)
        res_limit : float
            high-resolution limit in Angstrom
        """
        self.model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
        hkl_grid, self.map_shape = generate_grid(self.model.A_inv, 
                                                 self.hsampling, 
                                                 self.ksampling, 
                                                 self.lsampling,
                                                 return_hkl=True)
        self.q_grid = 2*np.pi*np.inner(self.model.A_inv.T, hkl_grid).T
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        self.mask, res_map = get_resolution_mask(self.model.cell, hkl_grid, res_limit)
    
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
    
    def apply_disorder(self, sigmas, num_rot=100, ensemble_dir=None):
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
            number of rotations to generate per sigma, or
            if ensemble_dir is not None, the nth ensemble member to generate
        ensemble_dir : str
            save unmasked structure factor amplitudes to given path

        Returns
        -------
        Id : numpy.ndarray, (n_sigma, q_grid.shape[0])
            diffuse intensity maps for the corresponding parameters
        """
        if type(sigmas) == float or type(sigmas) == int:
            sigmas = np.array([sigmas])

        if ensemble_dir is not None:
            out_prefix = f"rot_{num_rot:05}"
            num_rot=1
            
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
                    A = structure_factors(self.q_grid[self.mask], 
                                          xyz_rot[rnum], 
                                          self.model.ff_a[asu], 
                                          self.model.ff_b[asu], 
                                          self.model.ff_c[asu], 
                                          U=None, 
                                          batch_size=self.batch_size,
                                          n_processes=self.n_processes)
                    if ensemble_dir is not None:
                        np.save(os.path.join(ensemble_dir, out_prefix + f"_asu{asu}.npy"), A)
                    fc[self.mask] += A
                    fc_square[self.mask] += np.square(np.abs(A)) 
                Id[n_sigma] += fc_square / num_rot - np.square(np.abs(fc / num_rot))

        Id[:,~self.mask] = np.nan
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
    
class Ensemble:
    
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

class NonInteractingDeformableMolecules:

    """
    Lattice model with non-interacting deformable molecules.
    Each asymmetric unit is a Gaussian Network Model.
    """

    def __init__(self, pdb_path, hsampling, ksampling, lsampling,
                 expand_p1=True, res_limit=0, gnm_cutoff=4.,
                 batch_size=10000, n_processes=8):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.batch_size = batch_size
        self.n_processes = n_processes
        self._setup(pdb_path, expand_p1, res_limit)
        self._setup_gnm(pdb_path, gnm_cutoff)
        self._setup_covmat()
        
    def _setup(self, pdb_path, expand_p1, res_limit, q2_rounding=3):
        """
        Compute q-vectors to evaluate.

        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        res_limit : float
            high-resolution limit in Angstrom
        q2_rounding : int
            number of decimals to round q squared to, default: 3
        """
        self.model = AtomicModel(pdb_path, expand_p1=expand_p1)

        hkl_grid, self.map_shape = generate_grid(self.model.A_inv,
                                                 self.hsampling,
                                                 self.ksampling,
                                                 self.lsampling,
                                                 return_hkl=True)
        self.res_mask, res_map = get_resolution_mask(self.model.cell,
                                                     hkl_grid,
                                                     res_limit)
        self.q_grid = 2 * np.pi * np.inner(self.model.A_inv.T, hkl_grid).T

        q2 = np.linalg.norm(self.q_grid, axis=1) ** 2
        self.q2_unique, \
        self.q2_unique_inverse = np.unique(np.round(q2, q2_rounding),
                                           return_inverse=True)

    def _setup_gnm(self, pdb_path, gnm_cutoff):
        """
        Build Gaussian Network Model.

        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        gnm_cutoff : float
            distance cutoff used to define atom pairs
        """
        self.gnm = GaussianNetworkModel(pdb_path,
                                        enm_cutoff=gnm_cutoff,
                                        gamma_intra=1.,
                                        gamma_inter=0.)

    def _setup_covmat(self):
        """
        Compute the covariance matrix and perform low rank truncation.
        """
        self.compute_covariance_matrix()
        self.u, self.s = self._low_rank_truncation(self.covar)
        
    def compute_covariance_matrix(self):
        """
        Compute covariance matrix for one asymmetric unit.
        The covariance matrix results from modelling pairwise
        interactions with a Gaussian Network Model where atom
        pairs belonging to different asymmetric units are not
        interacting. It is scaled to match the ADPs in the input PDB file.
        """
        Kinv = self.gnm.compute_Kinv(self.gnm.compute_hessian())
        Kinv = np.real(Kinv[0, :, 0, :])  # only need that
        ADP_scale = np.mean(self.model.adp[0]) / \
                    (8 * np.pi * np.pi * np.mean(np.diag(Kinv)) / 3.)
        self.covar = Kinv * ADP_scale
        self.ADP = np.diag(self.covar)

    def _low_rank_truncation(self, sym_matrix, rank=None, rec_error_threshold=0.01):
        """
        Perform low rank approximation of a symmetric matrix.
        Either the desired rank is provided, or it is found
        when the reconstructed matrix Froebenius norm is the
        same as the original matrix, up to a provided
        reconstruction error.

        Parameters
        ----------
        sym_matrix : numpy.ndarray, shape (n, n)
            must be a 2D symmetric array
        rank : int or None
            if not None, the desired rank
        rec_error_threshold : float
            if rank is None, the maximal reconstruction error
        Returns
        -------
        u : numpy.ndarray, shape (n, rank)
            First rank-th components of sym_matrix
        s : numpy.ndarray, shape (rank,)
            First rank-th singular values of sym_matrix
        """
        u, s, vh = np.linalg.svd(sym_matrix)
        if rank is None:
            sym_matrix_norm = np.linalg.norm(sym_matrix)
            for rank in range(s.shape[0]):
                rec_matrix = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
                rec_matrix_norm = np.linalg.norm(rec_matrix)
                rec_error = 1. - rec_matrix_norm / sym_matrix_norm
                if rec_error < rec_error_threshold:
                    break
        return u[:, :rank], s[:rank]

    def compute_scl_intensity(self, rank=-1, outdir=None):
        """
        Compute diffuse intensity of non-interacting deformable
        molecules, in the soft-coupling limit.
        Namely, given a Gaussian Network Model (GNM) for the
        asymmetric unit (ASU), the covariance C is factorized:
        C = B * D @ B.T and eventually low-rank truncated.
        The diffuse intensity is then the incoherent sum over
        its components, weighted by q**2, where we introduce
        the component factors G = F * B:
        I(q) = q**2 \sum_r D_r \sum_asu |G_asu,r|**2

        Parameters
        ----------
        rank : int
            if -1, sum across ranks; else, save rank's results
        outdir : str
            path for storing rank results

        Returns
        -------
        Id : numpy.ndarray, shape (q_grid.shape[0],)
            diffuse intensity map
        """
        Id = np.zeros((self.q_grid.shape[0]))
        for i_asu in range(self.model.n_asu):
            if rank == -1:
                Id[self.res_mask] += np.dot(np.square(np.abs(structure_factors(self.q_grid[self.res_mask],
                                                                               self.model.xyz[i_asu],
                                                                               self.model.ff_a[i_asu],
                                                                               self.model.ff_b[i_asu],
                                                                               self.model.ff_c[i_asu],
                                                                               U=self.ADP,
                                                                               batch_size=self.batch_size,
                                                                               n_processes=self.n_processes,
                                                                               project_on_components=self.u,
                                                                               sum_over_atoms=False))), self.s)
            else:
                Id[self.res_mask] += np.square(np.abs(structure_factors(self.q_grid[self.res_mask],
                                                                        self.model.xyz[i_asu],
                                                                        self.model.ff_a[i_asu],
                                                                        self.model.ff_b[i_asu],
                                                                        self.model.ff_c[i_asu],
                                                                        U=self.ADP,
                                                                        batch_size=self.batch_size,
                                                                        n_processes=self.n_processes,
                                                                        project_on_components=self.u[:,rank],
                                                                        sum_over_atoms=False))) * self.s[rank]
        Id = np.multiply(self.q2_unique[self.q2_unique_inverse], Id)
        if outdir is not None:
            np.save(os.path.join(outdir, f"rank_{rank:05}.npy"), Id)
        return Id

    def compute_intensity_naive(self):
        """
        Compute diffuse intensity of non-interacting deformable
        molecules in a non-efficient / naive way.
        Namely, given a Gaussian Network Model (GNM) for the
        asymmetric unit (ASU), we can compute its covariance C
        and for each q and each atom pair T_ij(q) = exp(q**2 C_ij).
        Noting F_i(q) the structure factor for atom i, the
        contribution of one ASU to the intensity reads:
        I(q) = \sum_ij F_i(q) (T_ij(q) - 1.) F_j(q)
        The diffuse intensity is an incoherent sum over ASUs.
        """
        Id = np.zeros((self.q_grid.shape[0]), dtype='complex')

        self.compute_covariance_matrix()
        Tmat = np.exp(self.covar).astype(complex)

        F = np.zeros((self.model.n_asu,
                      self.q_grid.shape[0],
                      self.gnm.n_atoms_per_asu),
                     dtype='complex')
        for i_asu in range(self.model.n_asu):
            F[i_asu] = structure_factors(self.q_grid,
                                         self.model.xyz[i_asu],
                                         self.model.ff_a[i_asu],
                                         self.model.ff_b[i_asu],
                                         self.model.ff_c[i_asu],
                                         U=self.ADP,
                                         batch_size=self.batch_size,
                                         n_processes=self.n_processes,
                                         sum_over_atoms=False)

        for iq in tqdm(range(self.q_grid.shape[0])):
            Jq = np.power(Tmat,
                          self.q2_unique[self.q2_unique_inverse][iq]) - 1.
            for i_asu in range(self.model.n_asu):
                Id[iq] += np.matmul(F[i_asu,iq],
                                   np.matmul(Jq,
                                             np.conj(F[i_asu,iq])))
        return np.real(Id)

    def apply_disorder(self, scl=True):
        """
        Compute diffuse intensity of non-interacting deformable
        molecules.
        Namely, given a Gaussian Network Model (GNM) for the
        asymmetric unit (ASU), the covariance C between atomic
        displacements is computed. Then the coupling between
        structure factors F can be defined as J = exp(q2*C) - 1.
        The diffuse intensity is then the incoherent sum over
        asymmetric units: Id = \sum_asu F_asu.T J F_asu.
        Because computing this in the general case would not
        scale well, several approximations/simplifications are
        offered (at the moment, only the soft-coupling limit).

        Parameters
        ----------
        scl : bool
            whether we are in the soft-coupling limit or not.
        """
        if scl:
            Id = self.compute_scl_intensity()
        else:
            Id = self.compute_intensity_naive()
        return Id
