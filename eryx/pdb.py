import numpy as np
import gemmi

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
        self.structure = gemmi.read_pdb(pdb_file)
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
