import numpy as np
import scipy.signal
import scipy.spatial
from scipy.linalg import block_diag
import glob
import os
from tqdm import tqdm
from .pdb import AtomicModel, Crystal, GaussianNetworkModel
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

        self.scan_sigmas = []
        self.scan_ccs = []

    def plot_scan(self, output=None):
        """
        Plot results of scan, CC(sigma).

        Parameters
        ----------
        output : str
            if provided, save plot to given path
        """
        import matplotlib.pyplot as plt

        sigmas = np.array(self.scan_sigmas)
        ccs = np.array(self.scan_ccs)
        plt.scatter(sigmas, ccs, c='black')
        plt.plot(sigmas, ccs, c='black')
        plt.xlabel("$\sigma$ ($\mathrm{\AA}$)", fontsize=14)
        plt.ylabel("CC", fontsize=14)

        if output is not None:
            plt.savefig(output, dpi=300, bbox_inches='tight')
        
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
        self.scan_sigmas.extend(list(sigmas))
        self.scan_ccs.extend(list(ccs))
        
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
        self.opt_map = None
        
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
        if self.opt_map is None:
            self.opt_map = Id[opt_index].reshape(self.map_shape_nopad)
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
                U = self.model.adp[asu] / (8 * np.pi * np.pi)
                for rnum in range(num_rot):
                    A = structure_factors(self.q_grid[self.mask], 
                                          xyz_rot[rnum], 
                                          self.model.ff_a[asu], 
                                          self.model.ff_b[asu], 
                                          self.model.ff_c[asu], 
                                          U=U, 
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
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_p1=True, 
                 res_limit=0, batch_size=10000, n_processes=8, frame=-1):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1, res_limit, frame)
        self.batch_size = batch_size
        self.n_processes = n_processes
        
    def _setup(self, pdb_path, expand_p1, res_limit, frame):
        """
        Load model and compute q-vectors to evaluate / mask.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1 : bool
            if True, expand to p1 (i.e. if PDB corresponds to the asymmetric unit)
        res_limit : float
            high-resolution limit in Angstrom
        frame : int
            load specified conformation or all states if -1
        """
        self.model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=frame)
        hkl_grid, self.map_shape = generate_grid(self.model.A_inv, 
                                                 self.hsampling, 
                                                 self.ksampling, 
                                                 self.lsampling,
                                                 return_hkl=True)
        self.q_grid = 2*np.pi*np.inner(self.model.A_inv.T, hkl_grid).T
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        self.mask, res_map = get_resolution_mask(self.model.cell, hkl_grid, res_limit)
        self.frame = frame
        
    def apply_disorder(self, weights=None, ensemble_dir=None):
        """
        Compute the diffuse maps(s) resulting from ensemble disorder using
        Guinier's equation, and then taking the incoherent sum of all the
        asymmetric units. If an ensemble_dir is provided, the unweighted 
        results for the indicated state will be saved to disk.
        
        Parameters
        ----------
        weights : shape (n_sets, n_conf) 
            set(s) of probabilities associated with each conformation
        ensemble_dir : str
            save path for structure factor amplitudes for given state

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
                U = self.model.adp[index] / (8 * np.pi * np.pi)
                A = structure_factors(self.q_grid[self.mask], 
                                      self.model.xyz[index], 
                                      self.model.ff_a[index], 
                                      self.model.ff_b[index], 
                                      self.model.ff_c[index], 
                                      U=U, 
                                      batch_size=self.batch_size,
                                      n_processes=self.n_processes)

                if ensemble_dir is not None:
                    np.save(os.path.join(ensemble_dir, f"conf{self.frame:05}_asu{asu}.npy"), A)

                for nm in range(n_maps):
                    fc[nm][self.mask] += A * weights[nm][conf]
                    fc_square[nm][self.mask] += np.square(np.abs(A)) * weights[nm][conf]

            for nm in range(n_maps):
                Id[nm] += fc_square[nm] - np.square(np.abs(fc[nm]))
                
        Id[:,~self.mask] = np.nan
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
                    (8 * np.pi * np.pi * np.mean(np.diag(Kinv)))
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
        Id[~self.res_mask] = np.nan
        return Id

class OnePhonon:

    """
    Lattice of interacting rigid bodies in the one-phonon
    approximation (a.k.a small-coupling regime).
    """

    def __init__(self, pdb_path, hsampling, ksampling, lsampling,
                 expand_p1=True, group_by='asu',
                 res_limit=0., model='gnm',
                 gnm_cutoff=4., gamma_intra=1., gamma_inter=1.,
                 batch_size=10000, n_processes=8):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.batch_size = batch_size
        self.n_processes = n_processes
        self._setup(pdb_path, expand_p1, res_limit, group_by)
        self._setup_phonons(pdb_path, model,
                            gnm_cutoff, gamma_intra, gamma_inter)

    def _setup(self, pdb_path, expand_p1, res_limit, group_by):
        """
        Compute q-vectors to evaluate and build the unit cell
        and its nearest neighbors while storing useful dimensions.

        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1: bool
            expand_p1 : bool
            if True, expand to p1 (i.e. if PDB corresponds to the asymmetric unit)
        res_limit : float
            high-resolution limit in Angstrom
        group_by : str
            level of rigid-body assembly.
            For now, only None and 'asu' have been implemented.
        """
        self.model = AtomicModel(pdb_path, expand_p1)

        self.hkl_grid, self.map_shape = generate_grid(self.model.A_inv,
                                                      self.hsampling,
                                                      self.ksampling,
                                                      self.lsampling,
                                                      return_hkl=True)
        self.res_mask, res_map = get_resolution_mask(self.model.cell,
                                                     self.hkl_grid,
                                                     res_limit)
        self.q_grid = 2 * np.pi * np.inner(self.model.A_inv.T, self.hkl_grid).T

        self.crystal = Crystal(self.model)
        self.crystal.supercell_extent(nx=1, ny=1, nz=1)
        self.id_cell_ref = self.crystal.hkl_to_id([0,0,0])
        self.n_cell = self.crystal.n_cell
        self.n_asu = self.crystal.model.n_asu
        self.n_atoms_per_asu = self.crystal.get_asu_xyz().shape[0]
        self.n_dof_per_asu_actual = self.n_atoms_per_asu * 3

        self.group_by = group_by
        if self.group_by is None:
            self.n_dof_per_asu = np.copy(self.n_dof_per_asu_actual)
        else:
            self.n_dof_per_asu = 6
        self.n_dof_per_cell = self.n_asu * self.n_dof_per_asu

    def _setup_phonons(self, pdb_path, model,
                       gnm_cutoff, gamma_intra, gamma_inter):
        """
        Compute phonons either from a Gaussian Network Model of the
        molecules or by direct definition of the dynamical matrix.

        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        model : str
            chosen phonon model: 'gnm' or 'rb'
        gnm_cutoff : float
            distance cutoff used to define the GNM
            see eryx.pdb.GaussianNetworkModel.compute_hessian()
        gamma_intra: float
            spring constant for atom pairs belonging to the same molecule
            see eryx.pdb.GaussianNetworkModel.build_gamma()
        gamma_inter: float
            spring constant for atom pairs belonging to distinct molecules
            see eryx.pdb.GaussianNetworkModel.build_gamma()
        """
        self.kvec = np.zeros((self.hsampling[2],
                              self.ksampling[2],
                              self.lsampling[2],
                              3))
        self.kvec_norm = np.zeros((self.hsampling[2],
                                   self.ksampling[2],
                                   self.lsampling[2],
                                   1))
        self.V = np.zeros((self.hsampling[2],
                           self.ksampling[2],
                           self.lsampling[2],
                           self.n_asu * self.n_dof_per_asu,
                           self.n_asu * self.n_dof_per_asu),
                          dtype='complex')
        self.Winv = np.zeros((self.hsampling[2],
                              self.ksampling[2],
                              self.lsampling[2],
                              self.n_asu * self.n_dof_per_asu),
                             dtype='complex')

        self._build_A()
        self._build_M()
        self._build_kvec_Brillouin()
        if model == 'gnm':
            self._setup_gnm(pdb_path, gnm_cutoff, gamma_intra, gamma_inter)
            self.compute_gnm_phonons()
            self.compute_covariance_matrix()
        else:
            self.compute_rb_phonons()

    def _setup_gnm(self, pdb_path, gnm_cutoff, gamma_intra, gamma_inter):
        """
        Instantiate the Gaussian Network Model

        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        gnm_cutoff : float
            distance cutoff used to define the GNM
            see eryx.pdb.GaussianNetworkModel.compute_hessian()
        gamma_intra: float
            spring constant for atom pairs belonging to the same molecule
            see eryx.pdb.GaussianNetworkModel.build_gamma()
        gamma_inter: float
            spring constant for atom pairs belonging to distinct molecules
            see eryx.pdb.GaussianNetworkModel.build_gamma()
        """
        self.gnm = GaussianNetworkModel(pdb_path,
                                        enm_cutoff=gnm_cutoff,
                                        gamma_intra=gamma_intra,
                                        gamma_inter=gamma_inter)

    def _build_A(self):
        """
        Build the matrix A that projects small rigid-body displacements
        to the individual atoms in the rigid body.
        More specifically, consider the set of cartesian coordinates {r_i}_m
        of all atoms in the m-th rigid body and o_m their center of mass.
        Also consider their instantaneous displacement {u_i}_m translating
        from instantaneous rigid-body displacements w_m = [t_m, l_m] where
        t_m and l_m are respectively the 3-dimensional translation and libration
        vector of group m.
        For each atom i in group m, the conversion reads:
        u_i = A(r_i - o_m).w_m
        where A is the following 3x6 matrix:
        A(x,y,z) = [[ 1 0 0  0  z -y ]
                    [ 0 1 0 -z  0  x ]
                    [ 0 0 1  y -x  0 ]]
        """
        if self.group_by == 'asu':
            self.Amat = np.zeros((self.n_asu, self.n_atoms_per_asu, 3, 6))
            Atmp = np.zeros((3, 3))
            Adiag = np.copy(Atmp)
            np.fill_diagonal(Adiag, 1.)
            for i_asu in range(self.n_asu):
                xyz = np.copy(self.crystal.get_asu_xyz(i_asu))
                xyz -= np.mean(xyz, axis=0)
                for i_atom in range(self.n_atoms_per_asu):
                    Atmp[0, 1] = xyz[i_atom, 2]
                    Atmp[0, 2] = -xyz[i_atom, 1]
                    Atmp[1, 2] = xyz[i_atom, 0]
                    Atmp -= Atmp.T
                    self.Amat[i_asu, i_atom] = np.hstack([Adiag, Atmp])
            self.Amat = self.Amat.reshape((self.n_asu,
                                           self.n_dof_per_asu_actual,
                                           self.n_dof_per_asu))
        else:
            self.Amat = None

    def _build_M(self):
        """
        Build the mass matrix M.
        If all atoms are considered, M = M_0 is diagonal (see _build_M_allatoms())
        and Linv = 1./sqrt(M_0) is diagonal also.
        If atoms are grouped as rigid bodies, the all-atoms M matrix is
        projected using the A matrix: M = A.T M_0 A and Linv is obtained
        via Cholesky decomposition: M = LL.T
        """
        M_allatoms = self._build_M_allatoms()
        if self.group_by is None:
            M_allatoms = M_allatoms.reshape((self.n_asu * self.n_dof_per_asu_actual,
                                             self.n_asu * self.n_dof_per_asu_actual))
            self.Linv = 1. / np.sqrt(M_allatoms)
        else:
            Mmat = self._project_M(M_allatoms)
            Mmat = Mmat.reshape((self.n_asu * self.n_dof_per_asu,
                                 self.n_asu * self.n_dof_per_asu))
            self.Linv = np.linalg.inv(np.linalg.cholesky(Mmat))

    def _project_M(self, M_allatoms):
        """
        Project all-atom mass matrix M_0 using the A matrix: M = A.T M_0 A

        Parameters
        ----------
        M_allatoms : numpy.ndarray, shape (n_asu, n_atoms*3, n_asu, n_atoms*3)

        Returns
        -------
        Mmat: numpy.ndarray, shape (n_asu, n_dof_per_asu, n_asu, n_dof_per_asu)
        """
        Mmat = np.zeros((self.n_asu, self.n_dof_per_asu,
                         self.n_asu, self.n_dof_per_asu))
        for i_asu in range(self.n_asu):
            for j_asu in range(self.n_asu):
                Mmat[i_asu, :, j_asu, :] = \
                    np.matmul(self.Amat[i_asu].T,
                              np.matmul(M_allatoms[i_asu, :, j_asu, :],
                                        self.Amat[j_asu]))
        return Mmat

    def _build_M_allatoms(self):
        """
        Build all-atom mass matrix M_0

        Returns
        -------
        M_allatoms : numpy.ndarray, shape (n_asu, n_atoms*3, n_asu, n_atoms*3)
        """
        mass_array = np.array([element.weight for structure in self.crystal.model.elements for element in structure])
        mass_list = [np.kron(mass_array, np.eye(3))[:, 3 * i:3 * (i + 1)] for i in
                     range(self.n_asu * self.n_atoms_per_asu)]
        return block_diag(*mass_list).reshape((self.n_asu, self.n_dof_per_asu_actual,
                                               self.n_asu, self.n_dof_per_asu_actual))

    def _center_kvec(self, x, L):
        """
        For x and L integers such that 0 < x < L, return -L/2 < x < L/2
        by applying periodic boundary condition in L/2
        Parameters
        ----------
        x : int
            the index to center
        L : int
            length of the periodic box
        """
        return int(((x - L / 2) % L) - L / 2) / L

    def _build_kvec_Brillouin(self):
        """
        Compute all k-vectors and their norm in the first Brillouin zone.
        This is achieved by regularly sampling [-0.5,0.5[ for h, k and l.
        """
        for dh in range(self.hsampling[2]):
            k_dh = self._center_kvec(dh, self.hsampling[2])
            for dk in range(self.ksampling[2]):
                k_dk = self._center_kvec(dk, self.ksampling[2])
                for dl in range(self.lsampling[2]):
                    k_dl = self._center_kvec(dl, self.lsampling[2])
                    self.kvec[dh, dk, dl] = 2 * np.pi * np.inner(self.model.A_inv.T,
                                                                 (k_dh, k_dk, k_dl)).T
                    self.kvec_norm[dh, dk, dl] = np.linalg.norm(self.kvec[dh, dk, dl])

    def _at_kvec_from_miller_points(self, hkl_kvec):
        """
        Return the indices of all q-vector that are k-vector away from any
        Miller index in the map.

        Parameters
        ----------
        hkl_kvec : tuple of ints
            fractional Miller index of the desired k-vector
        """
        hsteps = int(self.hsampling[2] * (self.hsampling[1] - self.hsampling[0]) + 1)
        ksteps = int(self.ksampling[2] * (self.ksampling[1] - self.ksampling[0]) + 1)
        lsteps = int(self.lsampling[2] * (self.lsampling[1] - self.lsampling[0]) + 1)

        index_grid = np.mgrid[
                     hkl_kvec[0]:hsteps:self.hsampling[2],
                     hkl_kvec[1]:ksteps:self.ksampling[2],
                     hkl_kvec[2]:lsteps:self.lsampling[2]]

        return np.ravel_multi_index((index_grid[0].flatten(),
                                     index_grid[1].flatten(),
                                     index_grid[2].flatten()),
                                    self.map_shape)

    def _kvec_map(self):
        """
        Build a map where the intensity at each fractional Miller index
        is set as the norm of the corresponding k-vector.
        For example, at each integral Miller index, k-vector is zero and
        it increases and then decreases as we sample between them.

        Returns
        -------
        map : numpy.ndarray, shape (npoints, 1)
        """
        map = np.zeros((self.q_grid.shape[0]))
        for dh in tqdm(range(self.hsampling[2])):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                    map[q_indices] = np.linalg.norm(self.kvec[dh, dk, dl])
        return map

    def _q_map(self):
        """
        Build a map where the intensity at each fractional Miller index
        is set as the norm of the corresponding q-vector.

        Returns
        -------
        map : numpy.ndarray, shape (npoints, 1_
        """
        map = np.zeros((self.q_grid.shape[0]))
        for dh in tqdm(range(self.hsampling[2])):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                    map[q_indices] = np.linalg.norm(self.q_grid[q_indices], axis=1)
        return map

    def compute_hessian(self):
        """
        Build the projected Hessian matrix for the supercell.

        Returns
        -------
        hessian : numpy.ndarray,
                  shape (n_asu, n_dof_per_asu, n_cell, n_asu, n_dof_per_asu),
                  dtype 'complex'
            Hessian matrix for the assembly of rigid bodies in the supercell.
        """
        hessian = np.zeros((self.n_asu, self.n_dof_per_asu,
                            self.n_cell, self.n_asu, self.n_dof_per_asu),
                           dtype='complex')

        hessian_allatoms = self.gnm.compute_hessian()

        for i_cell in range(self.n_cell):
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    hessian[i_asu, :, i_cell, j_asu, :] = \
                        np.matmul(self.Amat[i_asu].T,
                                  np.matmul(np.kron(hessian_allatoms[i_asu, :, i_cell, j_asu, :],
                                                    np.eye(3)),
                                            self.Amat[j_asu]))

        return hessian

    def compute_covariance_matrix(self):
        """
        Compute covariance matrix for all asymmetric units.
        The covariance matrix results from modelling pairwise
        interactions with a Gaussian Network Model where atom
        pairs belonging to different asymmetric units are not
        interacting. It is scaled to match the ADPs in the input PDB file.
        """
        self.covar = np.zeros((self.n_asu*self.n_dof_per_asu,
                               self.n_cell, self.n_asu*self.n_dof_per_asu),
                              dtype='complex')

        hessian = self.compute_hessian()
        for dh in range(self.hsampling[2]):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    kvec = self.kvec[dh,dk,dl]
                    Kinv = self.gnm.compute_Kinv(hessian, kvec=kvec, reshape=False)
                    for j_cell in range(self.n_cell):
                        r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
                        phase = np.dot(kvec, r_cell)
                        eikr = np.cos(phase) + 1j * np.sin(phase)
                        self.covar[:,j_cell,:] += Kinv * eikr
        #ADP_scale = np.mean(self.model.adp[0]) / \
        #            (8 * np.pi * np.pi * np.mean(np.diag(self.covar[:,self.crystal.hkl_to_id([0,0,0]),:])) / 3.)
        #self.covar *= ADP_scale
        self.ADP = np.real(np.diag(self.covar[:,self.crystal.hkl_to_id([0,0,0]),:]))
        Amat = np.transpose(self.Amat, (1,0,2)).reshape(self.n_dof_per_asu_actual, self.n_asu*self.n_dof_per_asu)
        self.ADP = Amat @ self.ADP
        self.ADP = np.sum(self.ADP.reshape(int(self.ADP.shape[0]/3),3),axis=1)
        ADP_scale = np.mean(self.model.adp) / (8*np.pi*np.pi*np.mean(self.ADP)/3)
        self.ADP *= ADP_scale
        self.covar *= ADP_scale
        self.covar = np.real(self.covar.reshape((self.n_asu, self.n_dof_per_asu,
                                                 self.n_cell, self.n_asu, self.n_dof_per_asu)))

    def compute_gnm_phonons(self):
        """
        Compute the dynamical matrix for each k-vector in the first
        Brillouin zone, from the supercell's GNM.
        The squared inverse of their eigenvalues is
        stored for intensity calculation and their eigenvectors are
        mass-weighted to be used in the definition of the phonon
        structure factors.
        """
        hessian = self.compute_hessian()
        for dh in range(self.hsampling[2]):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    Kmat = self.gnm.compute_K(hessian, kvec=self.kvec[dh, dk, dl])
                    Kmat = Kmat.reshape((self.n_asu * self.n_dof_per_asu,
                                         self.n_asu * self.n_dof_per_asu))
                    Dmat = np.matmul(self.Linv,
                                     np.matmul(Kmat, self.Linv.T))
                    v, w, _ = np.linalg.svd(Dmat)
                    w = np.sqrt(w)
                    w = np.where(w < 1e-6, np.nan, w)
                    w = w[::-1]
                    v = v[:,::-1]
                    self.Winv[dh, dk, dl] = 1. / w ** 2
                    self.V[dh, dk, dl] = np.matmul(self.Linv.T, v)

    def compute_rb_phonons(self):
        """
        Compute the dynamical matrix for each k-vector in the first
        Brillouin zone as a decaying Gaussian of k.
        (in development, not fully tested or understood).
        """
        Kmat = np.zeros((self.n_asu * self.n_dof_per_asu,
                         self.n_asu * self.n_dof_per_asu))
        for dh in range(self.hsampling[2]):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    np.fill_diagonal(Kmat,
                                     np.exp(-0.5 * (
                                             np.linalg.norm(self.kvec[dh, dk, dl] /
                                                            np.linalg.norm(self.kvec[3, 0, 0])) ** 2))
                                     )
                    u, s, _ = np.linalg.svd(Kmat)
                    self.Winv[dh, dk, dl] = s
                    self.V[dh, dk, dl] = u

    def apply_disorder(self, rank=-1, outdir=None, use_data_adp=False):
        """
        Compute the diffuse intensity in the one-phonon scattering
        disorder model originating from a Gaussian Network Model
        representation of the asymmetric units, optionally reduced
        to a set of interacting rigid bodies.
        """
        if use_data_adp:
            ADP = self.model.adp[0] / (8 * np.pi * np.pi)
        else:
            ADP = self.ADP
        Id = np.zeros((self.q_grid.shape[0]), dtype='complex')
        for dh in tqdm(range(self.hsampling[2])):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):

                    q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                    q_indices = q_indices[self.res_mask[q_indices]]

                    F = np.zeros((q_indices.shape[0],
                                  self.n_asu,
                                  self.n_dof_per_asu),
                                 dtype='complex')
                    for i_asu in range(self.n_asu):
                        F[:, i_asu, :] = structure_factors(
                            self.q_grid[q_indices],
                            self.model.xyz[i_asu],
                            self.model.ff_a[i_asu],
                            self.model.ff_b[i_asu],
                            self.model.ff_c[i_asu],
                            U=ADP,
                            batch_size=self.batch_size,
                            n_processes=self.n_processes,
                            compute_qF=True,
                            project_on_components=self.Amat[i_asu],
                            sum_over_atoms=False)
                    F = F.reshape((q_indices.shape[0],
                                   self.n_asu * self.n_dof_per_asu))

                    if rank == -1:
                        Id[q_indices] += np.dot(
                            np.square(np.abs(np.dot(F, self.V[dh, dk, dl]))),
                            self.Winv[dh, dk, dl])
                    else:
                        Id[q_indices] += np.square(
                            np.abs(np.dot(F, self.V[dh,dk,dl,:,rank]))) * \
                                         self.Winv[dh,dk,dl,rank]
        Id[~self.res_mask] = np.nan
        Id = np.real(Id)
        if outdir is not None:
            np.save(os.path.join(outdir, f"rank_{rank:05}.npy"), Id)
        return Id

class OnePhononBrillouin:

    def __init__(self, pdb_path, h, k, l, N,
                 group_by='asu', model='gnm',
                 gnm_cutoff=4., gamma_intra=1., gamma_inter=1.,
                 batch_size=10000, n_processes=8):
        self.phonon = OnePhonon(pdb_path,(h-1,h+1,N),(k-1,k+1,N), (l-1,l+1,N),
                                group_by=group_by, model=model,
                                gnm_cutoff=gnm_cutoff, gamma_intra=gamma_intra, gamma_inter=gamma_inter,
                                batch_size=batch_size, n_processes=n_processes)
        self.Id = self.phonon.apply_disorder().reshape(self.phonon.map_shape)[N//2+1:-N//2,N//2+1:-N//2,N//2+1:-N//2]

