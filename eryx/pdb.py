import numpy as np
import gemmi
from scipy.spatial import KDTree

def sym_str_as_matrix(sym_str):
    """
    Comvert a symmetry operation from string to matrix format,
    only retaining the rotational component.
    
    Parameters
    ----------
    sym_str : str
        symmetry operation in string format, e.g. '-Y,X-Y,Z+1/3'
    
    Returns
    -------
    sym_matrix : numpy.ndarray, shape (3,3)
        rotation portion of symmetry operation in matrix format
    """
    sym_matrix = np.zeros((3,3))
    for i,item in enumerate(sym_str.split(",")):
        if '-X' in item:
            sym_matrix[i][0] = -1
        if 'X' in item and '-X' not in item:
            sym_matrix[i][0] = 1
        if 'Y' in item and '-Y' not in item:
            sym_matrix[i][1] = 1
        if '-Y' in item:
            sym_matrix[i][1] = -1
        if 'Z' in item and '-Z' not in item:
            sym_matrix[i][2] = 1
        if '-Z' in item:
            sym_matrix[i][2] = -1           
    return sym_matrix

def extract_sym_ops(pdb_file):
    """
    Extract the rotational component of the symmetry operations from 
    the PDB header (REMARK 290).

    Parameters
    ----------
    pdb_file : str
        path to coordinates file
    
    Returns
    -------
    sym_ops : dict
        dictionary of rotational component of symmetry operations
    """
    sym_ops = {}
    counter = 0
    with open(pdb_file, "r") as f:
        for line in f:
            if "REMARK 290" in line:
                if "555" in line:
                    sym_ops[counter] = sym_str_as_matrix(line.split()[-1])
                    counter += 1
            if "REMARK 300" in line:
                break
                
    return sym_ops

def extract_transformations(pdb_file):
    """
    Extract the transformations from the PDB header (REMARK 290).
    The rotational component of the matrices contain information
    about cell orthogonalization, import for non-orthogonal cells.
    The translations are in Angstroms rather than fractional cells.

    Parameters
    ----------
    pdb_file : str
        path to coordinates file
    
    Returns
    -------
    transformations : dict
        dictionary of symmetry operations as 3x4 matrices
    """
    transformations = {}
    with open(pdb_file, "r") as f:
        for line in f:
            if "REMARK 290" in line:
                if "SMTRY" in line:
                    sym_info = line.split()[3:]
                    op_num = int(sym_info[0]) - 1
                    op_line = np.array(sym_info[1:]).astype(float)
                    if op_num not in transformations:
                        transformations[op_num] = op_line
                    else:
                        transformations[op_num] = np.vstack((transformations[op_num], op_line))
            if "REMARK 300" in line:
                break
                
    return transformations

def get_unit_cell_axes(cell):
    """
    Compute axes for the unit cell.
    
    Parameters
    ----------
    cell : numpy.ndarray, shape (6,)
        unit cell parameters in Angstrom / degrees
        
    Returns
    -------
    axes : numpy.ndarray, shape (3,3)
        matrix of unit cell axes
    """
    a,b,c = cell[:3]
    alpha, beta, gamma = np.deg2rad(cell[3:])
    factor_21 = (np.cos(alpha) - np.cos(beta) * np.cos(gamma))/np.sin(gamma)
    factor_22 = np.sqrt(1 - np.square(np.cos(beta)) - np.square(factor_21))
    axes = np.array([[a, 0, 0],
                     [b * np.cos(gamma), b * np.sin(gamma), 0],
                     [c * np.cos(beta), c * factor_21, c*factor_22]])
    return axes

class AtomicModel:
    
    def __init__(self, pdb_file, expand_p1=False, frame=0, clean_pdb=True):
        self._get_gemmi_structure(pdb_file, clean_pdb)
        self._extract_cell()
        self.sym_ops, self.transformations = self._get_sym_ops(pdb_file)
        self.extract_frame(frame=frame, expand_p1=expand_p1)

    def _get_gemmi_structure(self, pdb_file, clean_pdb):
        """
        Retrieve Gemmi structure from PDB file.
        Optionally clean up water molecules, hydrogen, ...
        """
        self.structure = gemmi.read_pdb(pdb_file)
        if clean_pdb:
            self.structure.remove_alternative_conformations()
            self.structure.remove_hydrogens()
            self.structure.remove_waters()
            self.structure.remove_ligands_and_waters()
            self.structure.remove_empty_chains()

    def _extract_cell(self):
        """
        Extract unit cell information.
        """
        self.cell = np.array(self.structure.cell.parameters)
        self.A_inv = np.array(self.structure.cell.fractionalization_matrix)
        self.space_group = self.structure.spacegroup_hm
        self.unit_cell_axes = get_unit_cell_axes(self.cell)
        
    def _get_sym_ops(self, pdb_file):
        """
        Extract symmetry operations, preferably from the PDB
        header or alternatively from Gemmi. Since the latter
        has rotation operators that are unity, this may not 
        work correctly for non-orthongal cells.
    
        Parameters
        ----------
        pdb_file : str
            path to coordinates file in PDB format
        """
        sym_ops = extract_sym_ops(pdb_file)
        transformations = extract_transformations(pdb_file)
        
        if len(sym_ops) == 0:
            print(""""Warning: gathering symmetry operations from
            Gemmi rather than the PDB header. This may be incorrect
            for non-orthogonal unit cells.""")
            sym_ops = {}
            sg = gemmi.SpaceGroup(self.space_group) 
            for i,op in enumerate(sg.operations()):
                r = np.array(op.rot) / op.DEN 
                t = np.array(op.tran) / op.DEN
                sym_ops[i] = np.hstack((r, t[:,np.newaxis]))

        if len(transformations) == 0:
            transformations = sym_ops

        self.n_asu = len(transformations)
        return sym_ops, transformations
    
    def extract_frame(self, expand_p1=False, frame=0):
        """
        Extract the form factors and atomic coordinates for the
        given frame/model in the PDB, or if frame==-1, then for 
        all frames/models in the PDB. 
        
        Parameters
        ----------
        expand_p1 : bool
            if True, expand to all asymmetric units
        frame : int
            pdb frame to extract; if -1, extract all frames
        """
        self.ff_a, self.ff_b, self.ff_c = None, None, None
        self.adp = None
        self.elements = []
        self.xyz = None

        if frame == -1:
            frange = range(len(self.structure))
        else:
            frange = [frame]
        self.n_conf = len(frange)
            
        for fr in frange:
            model = self.structure[fr]
            residues = [res for ch in model for res in ch]
            self._extract_xyz(residues, expand_p1)
            self._extract_adp(residues, expand_p1)
            self._extract_ff_coefs(residues, expand_p1)
        
    def _get_xyz_asus(self, xyz):
        """
        Apply symmetry operations to get xyz coordinates of all 
        asymmetric units and pack them into the unit cell.
        
        Parameters
        ----------
        xyz : numpy.ndarray, shape (n_atoms, 3)
            atomic coordinates for a single asymmetric unit
            
        Returns
        -------
        xyz_asus : numpy.ndarray, shape (n_asu, n_atoms, 3)
            atomic coordinates for all asus in the unit cell
        """
        xyz_asus = []      
        for i,op in self.transformations.items():
            rot, trans = op[:3,:3], op[:,3]
            xyz_asu = np.inner(rot, xyz).T + trans
            com = np.mean(xyz_asu.T, axis=1)
            shift = np.sum(np.dot(self.unit_cell_axes.T, self.sym_ops[i]), axis=1)
            xyz_asu += np.abs(shift) * np.array(com < 0).astype(int)
            xyz_asu -= np.abs(shift) * np.array(com > self.cell[:3]).astype(int)
            xyz_asus.append(xyz_asu)
        
        return np.array(xyz_asus)

    def _extract_xyz(self, residues, expand_p1=False):
        """
        Extract atomic coordinates, optionally getting coordinates
        for all asymmetric units in the unit cell.
        
        Parameters
        ----------
        residues : list of gemmi.Residue objects
            residue objects containing atomic position information
        expand_p1 : bool
            if True, retrieve coordinates for all asus
            
        Returns
        -------
        xyz : numpy.ndarray, shape (n_atoms, 3) or (n_asu, n_atoms, 3)
            atomic coordinates in Angstroms
        """
        xyz = np.array([[atom.pos.tolist() for res in residues for atom in res]])
        if expand_p1:
            xyz = self._get_xyz_asus(xyz[0])
        
        if self.xyz is None:
            self.xyz = xyz
        else:
            self.xyz = np.vstack((self.xyz, xyz))

    def _extract_adp(self, residues, expand_p1):
        """
        Retrieve atomic displacement parameters.

        Parameters
        ----------
        residues : list of gemmi.Residue objects
            residue objects containing atomic position information
        expand_p1 : bool
            if True, expand to all asymmetric units
        """
        adp = np.array([[atom.b_iso for res in residues for atom in res]])

        if self.adp is None:
            self.adp = adp
        else:
            self.adp = np.vstack((self.adp, adp))

        if expand_p1:
            n_asu = self.xyz.shape[0]
            self.adp = np.tile(self.adp, (n_asu, 1))
    
    def _extract_ff_coefs(self, residues, expand_p1):
        """
        Retrieve atomic form factor coefficients, specifically
        a, b, and c of the following equation:
        f(qo) = sum(a_i * exp(-b_i * (qo/(4*pi))^2)) + c
        and the sum is over the element's coefficients. Also 
        store the gemmi element objects, an alternative way of
        computing the atomic form factors.
        
        Parameters
        ----------
        residues : list of gemmi.Residue objects
            residue objects containing atomic position information
        expand_p1 : bool
            if True, expand to all asymmetric units
        """
        ff_a = np.array([[atom.element.it92.a for res in residues for atom in res]])
        ff_b = np.array([[atom.element.it92.b for res in residues for atom in res]])
        ff_c = np.array([[atom.element.it92.c for res in residues for atom in res]])
        elements = [atom.element for res in residues for atom in res]
        
        if self.ff_a is None:
            self.ff_a, self.ff_b, self.ff_c = ff_a, ff_b, ff_c
        else:
            self.ff_a = np.vstack((self.ff_a, ff_a))
            self.ff_b = np.vstack((self.ff_b, ff_b))
            self.ff_c = np.vstack((self.ff_c, ff_c))
        self.elements.append(elements)
        
        if expand_p1:
            self._tile_form_factors()
    
    def flatten_model(self):
        """
        Set self variables to correspond to the given frame,
        for instance flattening the atomic coordinates from 
        (n_frames, n_atoms, 3) to (n_atoms, 3). 
        """
        n_asu = self.xyz.shape[0]
        self.xyz = self.xyz.reshape(-1, self.xyz.shape[-1])
        self.ff_a = self.ff_a.reshape(-1, self.ff_a.shape[-1])
        self.ff_b = self.ff_b.reshape(-1, self.ff_b.shape[-1])
        self.ff_c = self.ff_c.flatten()
        self.elements = [item for sublist in self.elements for item in sublist]
        
    def _tile_form_factors(self):
        """
        Tile the form factor coefficients and gemmi elements
        to match the number of asymmetric units.
        """
        n_asu = self.xyz.shape[0]
        self.ff_a = np.tile(self.ff_a, (n_asu, 1, 1))
        self.ff_b = np.tile(self.ff_b, (n_asu, 1, 1))
        self.ff_c = np.tile(self.ff_c, (n_asu, 1))
        self.elements = n_asu * self.elements


class Crystal:

    def __init__(self, atomic_model):
        """
        Parameters
        ----------
        atomic_model : an eryx.models.AtomicModel object
        """
        self.model = atomic_model
        self.supercell_extent()

    def supercell_extent(self, nx=0, ny=0, nz=0):
        """
        Define supercell dimensions. There will be nx cells on
        each +X and -X side of the reference cell, for a total
        of 2*nx + 1 cells along the X dimension. Same logic
        follows for the two other dimensions.

        Parameters
        ----------
        nx : integer
        ny : integer
        nz : integer
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.xrange = np.arange(-self.nx, self.nx + 1, 1)
        self.yrange = np.arange(-self.ny, self.ny + 1, 1)
        self.zrange = np.arange(-self.nz, self.nz + 1, 1)
        self.n_cell = (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1)

    def get_unitcell_origin(self, unit_cell=None):
        """
        Convert unit cell indices to spatial location of its origin.

        Parameters
        ----------
        unit_cell : list of 3 integer indices. Default: [0,0,0].
            index of the unit cell along the 3 dimensions.

        Returns
        -------
        origin : numpy.ndarray, shape (3,)
            location of the unit cell origin (in Angstrom)
        """
        if unit_cell is None:
            unit_cell = [0, 0, 0]
        origin = np.zeros(3)
        for i in range(3):
            origin += unit_cell[i] * self.model.unit_cell_axes[i]
        return origin

    def hkl_to_id(self, unit_cell=None):
        """
        Return unit cell index given its indices in the supercell.

        Parameters
        ----------
        unit_cell : list of 3 integer indices. Default: [0,0,0].
            index of the unit cell along the 3 dimensions.

        Returns
        -------
        cell_id : integer
            index of the unit cell in the supercell
        """
        if unit_cell is None:
            unit_cell = [0, 0, 0]
        icell = 0
        for h in self.xrange:
            for k in self.yrange:
                for l in self.zrange:
                    if h == unit_cell[0] and k == unit_cell[1] and l == unit_cell[2]:
                        cell_id = icell
                    icell += 1
        return cell_id

    def id_to_hkl(self, cell_id=0):
        """
        Return unit cell indices in supercell given its index.

        Parameters
        ----------
        cell_id : integer. Default: 0.

        Returns
        -------
        unit_cell : list of 3 integer indices.
            Index of the unit cell along the 3 dimensions.

        """
        icell = 0
        for h in self.xrange:
            for k in self.yrange:
                for l in self.zrange:
                    if icell == cell_id:
                        unit_cell = [h, k, l]
                    icell += 1
        return unit_cell

    def get_asu_xyz(self, asu_id=0, unit_cell=None):
        """

        Parameters
        ----------
        asu_id : integer. Default: 0.
            Asymmetric unit index.
        unit_cell : list of 3 integer indices.
            Index of the unit cell along the 3 dimensions.

        Returns
        -------
        xyz : numpy.ndarray, shape (natoms, 3)
            atomic coordinate for this asu in given unit cell.

        """
        if unit_cell is None:
            unit_cell = [0, 0, 0]
        xyz = self.model._get_xyz_asus(self.model.xyz[0])[asu_id]  # get asu
        xyz += self.get_unitcell_origin(unit_cell)  # move to unit cell
        return xyz

class GaussianNetworkModel:
    def __init__(self, pdb_path, enm_cutoff, gamma_intra, gamma_inter):
        self._setup_atomic_model(pdb_path)
        self.enm_cutoff = enm_cutoff
        self.gamma_inter = gamma_inter
        self.gamma_intra = gamma_intra
        self._setup_gaussian_network_model()

    def _setup_atomic_model(self, pdb_path):
        """
        Build unit cell and its nearest neighbors while
        storing useful dimensions.

        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        """
        atomic_model = AtomicModel(pdb_path, expand_p1=True)
        self.crystal = Crystal(atomic_model)
        self.crystal.supercell_extent(nx=1, ny=1, nz=1)
        self.id_cell_ref = self.crystal.hkl_to_id([0,0,0])
        self.n_cell = self.crystal.n_cell
        self.n_asu = self.crystal.model.n_asu
        self.n_atoms_per_asu = self.crystal.get_asu_xyz().shape[0]
        self.n_dof_per_asu_actual = self.n_atoms_per_asu * 3

    def _setup_gaussian_network_model(self):
        """
        Build interaction pair list and spring constant.
        """
        self.build_gamma()
        self.build_neighbor_list()

    def build_gamma(self):
        """
        The spring constant gamma dictates the interaction strength
        between pairs of atoms. It is defined across the main unit
        cell and its neighbors, for each asymmetric unit, resulting
        in an array of shape (n_cell, n_asu, n_asu).
        The spring constant between atoms belonging to different asus
        can be set to a different value than that for intra-asus atoms.
        """
        self.gamma = np.zeros((self.n_cell, self.n_asu, self.n_asu))
        for i_asu in range(self.n_asu):
            for i_cell in range(self.n_cell):
                for j_asu in range(self.n_asu):
                    self.gamma[i_cell, i_asu, j_asu] = self.gamma_inter
                    if (i_cell == self.id_cell_ref) and (j_asu == i_asu):
                        self.gamma[i_cell, i_asu, j_asu] = self.gamma_intra

    def build_neighbor_list(self):
        """
        Returns the list asu_neighbors[i_asu][i_cell][j_asu]
        of atom pairs between the ASU i_asu from the reference cell
        and ASU j_asu from the cell i_cell. The length of the returned
        list is the number of atoms in one ASU.
        For each atom i in the ASU i_asu in the reference cell,
        asu_neighbors[i_asu][i_cell][j_asu][i] returns the indices of
        its neighbors, if any, in the ASU j_asu in cell i_cell.
        """
        self.asu_neighbors = []

        for i_asu in range(self.n_asu):
            self.asu_neighbors.append([])
            kd_tree1 = KDTree(self.crystal.get_asu_xyz(i_asu, self.crystal.id_to_hkl(self.id_cell_ref)))

            for i_cell in range(self.n_cell):
                self.asu_neighbors[i_asu].append([])

                for j_asu in range(self.n_asu):
                    self.asu_neighbors[i_asu][i_cell].append([])
                    kd_tree2 = KDTree(self.crystal.get_asu_xyz(j_asu, self.crystal.id_to_hkl(i_cell)))

                    self.asu_neighbors[i_asu][i_cell][j_asu] = kd_tree1.query_ball_tree(kd_tree2, r=self.enm_cutoff)

    def compute_hessian(self):
        """
        For a pair of atoms the Hessian in a GNM is defined as:
        1. i not j and dij =< cutoff: -gamma_ij
        2. i not j and dij > cutoff: 0
        3. i=j: -sum_{j not i} hessian_ij

        Returns
        -------
        hessian: numpy.ndarray,
                 shape (n_asu, n_atoms_per_asu,
                        n_cell, n_asu, n_atoms_per_asu)
                 type 'complex'
            - dimension 0: index ASUs in reference cell
            - dimension 1: index their atoms
            - dimension 2: index neighbor cells
            - dimension 3: index ASUs in neighbor cell
            - dimension 4: index atoms in neighbor ASU
        """
        hessian = np.zeros((self.n_asu, self.n_atoms_per_asu,
                            self.n_cell, self.n_asu, self.n_atoms_per_asu),
                           dtype='complex')
        hessian_diagonal = np.zeros((self.n_asu, self.n_atoms_per_asu),
                                    dtype='complex')

        # off-diagonal
        for i_asu in range(self.n_asu):
            for i_cell in range(self.n_cell):
                for j_asu in range(self.n_asu):
                    for i_at in range(self.n_atoms_per_asu):
                        iat_neighbors = self.asu_neighbors[i_asu][i_cell][j_asu][i_at]
                        if len(iat_neighbors) > 0:
                            hessian[i_asu, i_at, i_cell, j_asu, iat_neighbors] = -self.gamma[i_cell, i_asu, j_asu]
                            hessian_diagonal[i_asu, i_at] -= self.gamma[i_cell, i_asu, j_asu] * len(iat_neighbors)

        # diagonal (also correct for over-counted self term)
        for i_asu in range(self.n_asu):
            for i_at in range(self.n_atoms_per_asu):
                hessian[i_asu, i_at, self.id_cell_ref, i_asu, i_at] = -hessian_diagonal[i_asu, i_at] - self.gamma[
                    self.id_cell_ref, i_asu, i_asu]

        return hessian

    def compute_K(self, hessian, kvec=None):
        """
        Noting H(d) the block of the hessian matrix
        corresponding the the d-th reference cell
        whose origin is located at r_d, then:
        K(kvec) = \sum_d H(d) exp(i kvec. r_d)

        Parameters
        ----------
        hessian : numpy.ndarray, see compute_hessian()
        kvec : numpy.ndarray, shape (3,)
            phonon wavevector, default array([0.,0.,0.])

        Returns
        -------
        Kmat : numpy.ndarray,
               shape (n_asu, n_atoms_per_asu,
                      n_asu, n_atoms_per_asu)
               type 'complex'
        """
        if kvec is None:
            kvec = np.zeros(3)
        Kmat = np.copy(hessian[:, :, self.id_cell_ref, :, :])

        for j_cell in range(self.n_cell):
            if j_cell == self.id_cell_ref:
                continue
            r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
            phase = np.dot(kvec, r_cell)
            eikr = np.cos(phase) + 1j * np.sin(phase)
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    Kmat[i_asu, :, j_asu, :] += hessian[i_asu, :, j_cell, j_asu, :] * eikr
        return Kmat

    def compute_Kinv(self, hessian, kvec=None, reshape=True):
        """
        Compute the inverse of K(kvec)
        (see compute_K() for the relationship between K and the hessian).

        Parameters
        ----------
        hessian : numpy.ndarray, see compute_hessian()
        kvec : numpy.ndarray, shape (3,)
            phonon wavevector, default array([0.,0.,0.])

        Returns
        -------
        Kinv : numpy.ndarray,
               shape (n_asu, n_atoms_per_asu,
                      n_asu, n_atoms_per_asu)
               type 'complex'
        """
        if kvec is None:
            kvec = np.zeros(3)
        Kmat = self.compute_K(hessian, kvec=kvec)
        Kshape = Kmat.shape
        Kinv = np.linalg.pinv(Kmat.reshape(Kshape[0] * Kshape[1],
                                           Kshape[2] * Kshape[3]))
        if reshape:
            Kinv = Kinv.reshape((Kshape[0], Kshape[1],
                                 Kshape[2], Kshape[3]))
        return Kinv
