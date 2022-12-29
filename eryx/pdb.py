import numpy as np
import gemmi

class AtomicModel:
    
    def __init__(self, pdb_file, expand_p1=False, frame=0):
        self.structure = gemmi.read_pdb(pdb_file)
        self.model = self.structure[frame]
        self.residues = [res for ch in self.model for res in ch]
        self._extract_cell()
        self.xyz = self._extract_xyz(expand_p1)
        self._extract_ff_coefs()
        
    def _extract_cell(self):
        """
        Extract unit cell information.
        """
        self.cell = np.array(self.structure.cell.parameters)
        self.A_inv = np.array(self.structure.cell.fractionalization_matrix)
        
    def _get_xyz_asus(self, xyz):
        """
        Apply symmetry operations to get xyz coordinates
        for all asymmetric units in the unit cell.

        Parameters
        ----------
        xyz : numpy.ndarray, shape (n_atoms, 3)
            atomic coordinates for a single asymmetric unit

        Returns
        -------
        xyz_asus : numpy.ndarray, shape (n_asu, n_atoms, 3)
            atomic coordinates for all asus in the unit cell
        """
        sg = gemmi.SpaceGroup(self.structure.spacegroup_hm)
        xyz_asus = []
        
        for op in sg.operations():
            rot = np.array(op.rot) / op.DEN
            trans = np.array(op.tran) / op.DEN 
            xyz_asu = np.inner(rot, xyz).T + trans * self.cell[:3]  
            com = np.mean(xyz_asu.T, axis=1)
            xyz_asu += self.cell[:3] * np.array(com < 0).astype(int)
            xyz_asus.append(xyz_asu)
        return np.array(xyz_asus)

    def _extract_xyz(self, expand_p1=False):
        """
        Extract atomic coordinates, optionally getting coordinates
        for all asymmetric units in the unit cell.
        
        Parameters
        ----------
        expand_p1 : bool
            if True, retrieve coordinates for all asus
            
        Returns
        -------
        xyz : numpy.ndarray, shape (n_atoms, 3) or (n_asu, n_atoms, 3)
            atomic coordinates in Angstroms
        """
        xyz = np.array([atom.pos.tolist() for res in self.residues for atom in res])
        if expand_p1:
            xyz = self._get_xyz_asus(xyz)
        return xyz
    
    def _extract_ff_coefs(self):
        """
        Retrieve atomic form factor coefficients, specifically
        a, b, and c of the following equation:
        f(qo) = sum(a_i * exp(-b_i * (qo/(4*pi))^2)) + c
        and the sum is over the element's coefficients. Also 
        store the gemmi element objects, an alternative way of
        computing the atomic form factors.
        """
        self.ff_a = np.array([atom.element.it92.a for res in self.residues for atom in res])
        self.ff_b = np.array([atom.element.it92.b for res in self.residues for atom in res])
        self.ff_c = np.array([atom.element.it92.c for res in self.residues for atom in res])
        self.elements = [atom.element for res in self.residues for atom in res]
        
    def concatenate_asus(self):
        """
        Flatten the xyz coordinates from dimensions (n_asu, n_asu_atoms, 3)
        to (n_asu * n_asu_atoms, 3), and tile the form factor coefficients
        and gemmi element objects to match.
        """
        n_asu = self.xyz.shape[0]
        self.xyz = self.xyz.reshape(-1, self.xyz.shape[-1])
        self.ff_a = np.tile(self.ff_a, (n_asu, 1))
        self.ff_b = np.tile(self.ff_b, (n_asu, 1))
        self.ff_c = np.tile(self.ff_c, 4)
        self.elements = n_asu * self.elements
