import numpy as np
import gemmi

def generate_grid(A_inv, hsampling, ksampling, lsampling, return_hkl=False):
    """
    Generate a grid of q-vectors based on the desired extents 
    and spacing in hkl space.
    
    Parameters
    ----------
    A_inv : numpy.ndarray, shape (3,3)
        fractional cell orthogonalization matrix
    hsampling : tuple, shape (3,)
        (hmin, hmax, oversampling relative to Miller indices)
    ksampling : tuple, shape (3,)
        (kmin, kmax, oversampling relative to Miller indices)
    lsampling : tuple, shape (3,)
        (lmin, lmax, oversampling relative to Miller indices)
    return_hkl : bool
        if True, return hkl indices rather than q-vectors
    
    Returns
    -------
    q_grid or hkl_grid : numpy.ndarray, shape (n_points, 3)
        grid of q-vectors or hkl indices
    map_shape : tuple, shape (3,)
        shape of 3d map
    """
    hsteps = int(hsampling[2]*(hsampling[1]-hsampling[0])+1)
    ksteps = int(ksampling[2]*(ksampling[1]-ksampling[0])+1)
    lsteps = int(lsampling[2]*(lsampling[1]-lsampling[0])+1)
    
    hkl_grid = np.mgrid[lsampling[0]:lsampling[1]:lsteps*1j,
                        ksampling[0]:ksampling[1]:ksteps*1j,
                        hsampling[0]:hsampling[1]:hsteps*1j]
    map_shape = hkl_grid.shape[1:][::-1]
    hkl_grid = hkl_grid.T.reshape(-1,3)
    hkl_grid = hkl_grid[:, [2,1,0]]
    
    if return_hkl:
        return hkl_grid, map_shape
    else:
        q_grid = 2*np.pi*np.inner(A_inv.T, hkl_grid).T
        return q_grid, map_shape

def get_symmetry_equivalents(hkl_grid, sym_ops):
    """
    Get symmetry equivalent Miller indices of input hkl_grid.
    The symmetry-equivalents are stacked horizontally, so that
    the first dimension of the output array corresponds to the
    nth asymmetric unit.
    
    Parameters
    ----------
    hkl_grid : numpy.ndarray, shape (n_points, 3)
        hkl indices corresponding to flattened intensity map
    sym_ops : dict
        rotational symmetry operations as 3x3 arrays
        
    Returns
    -------
    hkl_grid_sym : numpy.ndarray, shape (n_asu, n_points, 3)
        stacked hkl indices of symmetry-equivalents
    """
    hkl_grid_sym = np.empty(3)
    for i,rot in sym_ops.items():
        hkl_grid_rot = np.matmul(hkl_grid, rot)
        hkl_grid_sym = np.vstack((hkl_grid_sym, hkl_grid_rot))
    hkl_grid_sym = hkl_grid_sym[1:]
    return hkl_grid_sym.reshape(len(sym_ops), hkl_grid.shape[0], 3)
    
def get_ravel_indices(hkl_grid_sym, sampling):
    """
    Map 3d hkl indices to corresponding 1d indices after raveling.
    
    Parameters
    ----------
    hkl_grid_sym : numpy.ndarray, shape (n_asu, n_points, 3)
        stacked hkl indices of symmetry-equivalents
    sampling : tuple, shape (3,)
        sampling rate relative to integral Millers along (h,k,l)
    
    Returns
    -------
    ravel : numpy.ndarray, shape (n_asu, n_points)
        indices in raveled space for hkl_grid_sym
    """
    hkl_grid_stacked = hkl_grid_sym.reshape(-1, hkl_grid_sym.shape[-1])
    hkl_grid_int = np.around(hkl_grid_stacked * np.array(sampling)).astype(int)
    lbounds = np.min(hkl_grid_int, axis=0)
    ubounds = np.max(hkl_grid_int, axis=0)
    map_shape_ravel = tuple((ubounds - lbounds + 1)) 
    hkl_grid_int = hkl_grid_int.reshape(hkl_grid_sym.shape)
    
    ravel = np.zeros(hkl_grid_sym.shape[:2]).astype(int)
    for i in range(ravel.shape[0]):
        ravel[i] = np.ravel_multi_index((hkl_grid_int[i] - lbounds).T, map_shape_ravel)

    return ravel

def cos_sq(angles):
    """ Compute cosine squared of input angles in radians. """
    return np.square(np.cos(angles))

def sin_sq(angles):
    """ Compute sine squared of input angles in radianss. """
    return np.square(np.sin(angles))

def compute_resolution(cell, hkl):
    """
    Compute reflections' resolution in 1/Angstrom. To check, see: 
    https://www.ruppweb.org/new_comp/reciprocal_cell.htm.
        
    Parameters
    ----------
    cell : numpy.ndarray, shape (6,)
        unit cell parameters (a,b,c,alpha,beta,gamma) in Ang/deg
    hkl : numpy.ndarray, shape (n_refl, 3)
        Miller indices of reflections
            
    Returns
    -------
    resolution : numpy.ndarray, shape (n_refl)
        resolution associated with each reflection in Angstrom
    """

    a,b,c = [cell[i] for i in range(3)] 
    alpha,beta,gamma = [np.radians(cell[i]) for i in range(3,6)] 
    h,k,l = [hkl[:,i] for i in range(3)]

    pf = 1.0 - cos_sq(alpha) - cos_sq(beta) - cos_sq(gamma) + 2.0*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
    n1 = np.square(h)*sin_sq(alpha)/np.square(a) + np.square(k)*sin_sq(beta)/np.square(b) + np.square(l)*sin_sq(gamma)/np.square(c)
    n2a = 2.0*k*l*(np.cos(beta)*np.cos(gamma) - np.cos(alpha))/(b*c)
    n2b = 2.0*l*h*(np.cos(gamma)*np.cos(alpha) - np.cos(beta))/(c*a)
    n2c = 2.0*h*k*(np.cos(alpha)*np.cos(beta) - np.cos(gamma))/(a*b)

    return 1.0 / np.sqrt((n1 + n2a + n2b + n2c) / pf)

def get_hkl_extents(cell, resolution, oversampling=1):
    """
    Determine the min/max hkl for the given cell and resolution.
    
    Parameters
    ----------
    cell : numpy.ndarray, shape (6,)
        unit cell parameters in Angstrom / degrees
    resolution : float
        high-resolution limit
    oversampling : int or tuple of shape (3,)
        oversampling rate relative to integral Miller indices
    
    Returns
    -------
    hsampling : tuple, shape (3,)
        (min, max, interval) along h axis
    ksampling : tuple, shape (3,)
        (min, max, interval) along k axis
    lsampling : tuple, shape (3,)
        (min, max, interval) along l axis        
    """
    if type(oversampling) == int:
        oversampling = 3 * [oversampling]
    g_cell = gemmi.UnitCell(*cell)
    h,k,l = g_cell.get_hkl_limits(resolution)
    return (-h,h,oversampling[0]), (-k,k,oversampling[1]), (-l,l,oversampling[2])

def expand_sym_ops(sym_ops):
    """
    Expand symmetry operations to include Friedel equivalents.
    
    Parameters
    ----------
    sym_ops : dict
        rotational symmetry operations as 3x3 matrices
    
    Returns
    -------
    sym_ops_exp : dict
        sym_ops, expanded to account for Friedel symmetry
    """
    sym_ops_exp = dict(sym_ops)
    for key in sym_ops:
        sym_ops_exp[key + len(sym_ops)] = -1 * sym_ops[key]
    return sym_ops_exp

def compute_multiplicity(model, hsampling, ksampling, lsampling):
    """
    Compute the multiplicity of each voxel in the map.
    
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
    
    Returns
    -------
    hkl_grid : numpy.ndarray, shape (n_points, 3)
        grid of hkl vectors
    multiplicity : numpy.ndarray, 3d
        multiplicity of each grid point in the map
    """
    sym_ops_exp = expand_sym_ops(model.sym_ops)
    hkl_grid, map_shape = generate_grid(model.A_inv, hsampling, ksampling, lsampling, return_hkl=True)
    hkl_sym = get_symmetry_equivalents(hkl_grid, sym_ops_exp)
    ravel = get_ravel_indices(hkl_sym, (hsampling[2], ksampling[2], lsampling[2])).T
    multiplicity = (np.diff(np.sort(ravel,axis=1),axis=1)!=0).sum(axis=1)+1
    return hkl_grid, multiplicity.reshape(map_shape)

def parse_asu_condition(asu_condition):
    """
    Parse Gemmi's string describing which reflections belong to the 
    asymmetric unit into a string compatible with python's eval.
    
    Parameters
    ----------
    asu_condition : str
        Gemmi-style string describing asu
        
    Returns
    -------
    asu_condition : str
        eval-compatible string describing asu
    """
    # convert to numpy boolean operators
    find = ["=", "and", "or", ">==", "<=="] 
    replace = ["==", "&", "|", ">=", "<="]
    for i,j in zip(find, replace):
        asu_condition = asu_condition.replace(i,j)
        
    # add missing parenthesis around individual conditions
    alphas = [idx for idx in range(len(asu_condition)) if (asu_condition[idx].isalpha() and asu_condition[idx+1] not in [" ", ")", "("])]
    counter = 0
    for start in alphas:
        asu_condition = asu_condition[:start+counter] + "(" + asu_condition[start+counter:]
        counter += 1
        end = asu_condition.find(" ", start+counter)
        if end == -1:
            asu_condition = asu_condition + ")"
        else:
            asu_condition = asu_condition[:end] + ")" + asu_condition[end:]
        counter += 1
    
    return asu_condition

def get_asu_mask(space_group, hkl_grid):
    """
    Generate a boolean mask that indicates which hkl indices belong
    to the asymmetric unit.
    
    Parameters
    ----------
    space_group : int or str
        crystal's space group
    hkl_grid : numpy.ndarray, shape (n_points, 3)
        hkl indices 
        
    Returns
    -------
    asu_mask : numpy.ndarray, shape (n_points,)
        True indicates hkls that belong to the asymmetric unit
    """
    sg = gemmi.SpaceGroup(space_group)
    asu_condition = gemmi.ReciprocalAsu(sg).condition_str()
    asu_condition = parse_asu_condition(asu_condition)
    h, k, l = [hkl_grid[:,i] for i in range(3)]
    asu_mask = eval(asu_condition)
    return asu_mask
