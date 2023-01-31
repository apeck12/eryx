import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import numpy as np
import plotly.graph_objects as go

def visualize_central_slices(I, vmax_scale=5, contour=False):
    """
    Plot central slices from the input map,  assuming
    that the map is centered around h,k,l=(0,0,0).

    Parameters
    ----------
    I : numpy.ndarray, 3d
        intensity map
    vmax_scale : float
        vmax will be vmax_scale*mean(I)
    """
    f, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12,4))
    map_shape = I.shape
    vmax = I[~np.isnan(I)].mean()*vmax_scale

    if contour:
        ax1.contourf(I[int(map_shape[0] / 2), :, :], origin='upper')
        ax2.contourf(I[:, int(map_shape[1] / 2), :], origin='upper')
        ax3.contourf(I[:, :, int(map_shape[2] / 2)], origin='upper')

        ax1.set_title("View along h", fontsize=14)
        ax2.set_title("View along k", fontsize=14)
        ax3.set_title("View along l", fontsize=14)

    else:
        ax1.imshow(I[int(map_shape[0]/2),:,:], vmax=vmax)
        ax2.imshow(I[:,int(map_shape[1]/2),:], vmax=vmax)
        ax3.imshow(I[:,:,int(map_shape[2]/2)], vmax=vmax)

        ax1.set_title("(0,k,l)", fontsize=14)
        ax2.set_title("(h,0,l)", fontsize=14)
        ax3.set_title("(h,k,0)", fontsize=14)

    ax1.set_aspect(map_shape[2]/map_shape[1])
    ax2.set_aspect(map_shape[2]/map_shape[0])
    ax3.set_aspect(map_shape[1]/map_shape[0])


    for ax in [ax1,ax2,ax3]:
        ax.set_xticks([])
        ax.set_yticks([])


class VisualizeCrystal:

    def __init__(self, crystal, gnm=None, nidm=None):
        """
        Plotly helper functions to visualize the Crystal object.
        Parameters
        ----------
        crystal : eryx.pdb.Crystal object
        gnm : (optional) eryx.pdb.GaussianNetworkModel object
        nidm : (optional) eryx.models.NonInteractingDeformableMolecules object
        """
        self.crystal = crystal
        self.draw_data = None
        self.color_by = 'asu_id'
        self.color_palette = 'xkcd'
        self.gnm = gnm   # if not None, show inter ASU contacts
        if self.gnm is not None:
            self.gnm_contacts_indices = self._setup_gnm_contacts()
        self.nidm = nidm # if not None, show intra ASU covariances
        if self.nidm is not None:
            self.nidm_covar_indices = self._setup_nidm_covar()

    def show(self):
        """
        Show the crystal's supercell in a plotly figure.
        """
        self.draw_supercell()
        self.fig.show()

    def draw_supercell(self):
        """
        For every cell in the crystal' supercell,
        draw its cell axes and the asymmetric units it contains.
        """
        for h in self.crystal.xrange:
            for k in self.crystal.yrange:
                for l in self.crystal.zrange:
                    self.draw_unit_cell_axes(origin=self.crystal.get_unitcell_origin(unit_cell=[h,k,l]))
                    for i_asu in np.arange(self.crystal.model.n_asu):
                        self.draw_asu(i_asu, unit_cell=[h,k,l],
                                      name=f'Cell #{self.crystal.hkl_to_id(unit_cell=[h,k,l])} | ASU #{i_asu}')
                        if self.gnm is not None:
                            self._draw_network(self.gnm_contacts_indices,
                                               i_asu, unit_cell=[h,k,l])
                        if self.nidm is not None:
                            self._draw_network(self.nidm_covar_indices,
                                               i_asu, unit_cell=[h,k,l])

    def draw_unit_cell_axes(self, origin=np.array([0., 0., 0.]), showlegend=False):
        """
        Draw the axes of a cell as a Scatter3D plotly object.

        Parameters
        ----------
        origin : numpy.ndarray, shape (3,), default: np.array([0.,0.,0.])
            3d coordinate of the unit cell origin.
        showlegend : bool, default: False
            Whether the object appears in the legend or not.
        """
        u_xyz = self.crystal.model.unit_cell_axes
        p_xyz = np.zeros((16, 3))
        p_sign_sequence = [1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1]
        p_id_sequence = [2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 2, 1]
        p_xyz[0] = origin
        for i in range(15):
            p_xyz[i + 1] = p_xyz[i] + p_sign_sequence[i] * u_xyz[p_id_sequence[i]]
        self._draw_go_vector(self._build_go_vector(p_xyz,
                                                   line=dict(color="gray", width=1),
                                                   showlegend=showlegend))

    def draw_asu(self, asu_id=0, unit_cell=None, name='asu0', showlegend=True):
        """
        Draw an asymmetric unit as a Scatter3d plotly object.

        Parameters
        ----------
        asu_id : asymmetric unit index
        unit_cell : list of 3 integer indices.
            Index of the unit cell along the 3 dimensions.
        name : string, default: 'asu0'
            Name displayed in legend.
        showlegend : bool, default: True
            Whether the object appears in the legend or not
        """
        if unit_cell is None:
            unit_cell = [0, 0, 0]
        self._draw_go_vector(self._build_go_vector(self.crystal.get_asu_xyz(asu_id, unit_cell),
                                                   mode='markers',
                                                   marker=dict(size=5,
                                                               color=self._get_color(asu_id, unit_cell)),
                                                   name=name,
                                                   showlegend=showlegend))

    def _setup_gnm_contacts(self):
        indices = []
        for i_asu in range(self.crystal.model.n_asu):
            indices.append([])
            for j_cell in range(self.crystal.n_cell):
                indices[i_asu].append([])
                for j_asu in range(self.crystal.model.n_asu):
                    indices[i_asu][j_cell].append([])
                    if i_asu == j_asu and j_cell == self.crystal.hkl_to_id([0, 0, 0]):
                        indices[i_asu][j_cell][j_asu].append([])
                    else:
                        indices[i_asu][j_cell][j_asu] = self.gnm.asu_neighbors[i_asu][j_cell][j_asu]
        return indices

    def _setup_nidm_covar(self):
        threshold = np.mean(self.nidm.covar.flatten()) \
                    + 5*np.std(self.nidm.covar.flatten())
        asu_indices = []
        pairs = np.where(self.nidm.covar > threshold)
        for i in range(self.crystal.get_asu_xyz().shape[0]):
            i_indices = np.where(pairs[0]==i)
            j = pairs[1][i_indices]
            j_indices = np.where(pairs[1][i_indices]>i)[0]
            asu_indices.append(j[j_indices].tolist())

        indices = []
        for i_asu in range(self.crystal.model.n_asu):
            indices.append([])
            for j_cell in range(self.crystal.n_cell):
                indices[i_asu].append([])
                for j_asu in range(self.crystal.model.n_asu):
                    indices[i_asu][j_cell].append([])
                    if i_asu == j_asu and j_cell == self.crystal.hkl_to_id([0,0,0]):
                        indices[i_asu][j_cell][j_asu] = asu_indices
        return indices

    def _draw_network(self, indices, asu_id=0, unit_cell=None, showlegend=False):
        if unit_cell is None:
            unit_cell = [0, 0, 0]
        xyz = np.zeros((2, 3))
        for i_asu in range(self.crystal.model.n_asu):
            pairs = indices[i_asu][self.crystal.hkl_to_id(unit_cell)][asu_id]
            for i_at in range(len(pairs)):
                if len(pairs[i_at]) == 0:
                    continue
                for j_at in pairs[i_at]:
                    #print(i_at, j_at, i_asu, self.crystal.hkl_to_id(unit_cell), asu_id)
                    xyz[0] = self.crystal.get_asu_xyz(i_asu,[0,0,0])[i_at]
                    xyz[1] = self.crystal.get_asu_xyz(asu_id,unit_cell)[j_at]
                    self._draw_go_vector(
                        self._build_go_vector(xyz=xyz,
                                              line=dict(color=self._get_color(asu_id,unit_cell)),
                                              showlegend=showlegend))

    def _draw_go_vector(self, go_vector):
        if self.draw_data is None:
            self.draw_data = [go_vector]
        else:
            self.draw_data.append(go_vector)
        self.fig = go.Figure(data=self.draw_data)

    def _build_go_vector(self, xyz, mode='lines', line=None, marker=None,
                         name=None, showlegend=True):
        if line is None:
            line = {}
        if marker is None:
            marker = {}
        return go.Scatter3d(x=xyz[:, 0],
                            y=xyz[:, 1],
                            z=xyz[:, 2],
                            mode=mode,
                            line=line,
                            marker=marker,
                            name=name,
                            showlegend=showlegend)

    def _get_color(self, asu_id, unit_cell=None):
        """
        Return the ASU color.

        Parameters
        ----------
        asu_id : asymmetric unit index
        unit_cell : list of 3 integer indices.
            Index of the unit cell along the 3 dimensions.
        """
        if unit_cell is None:
            unit_cell = [0, 0, 0]
        idx = 0
        ndx = 1
        if self.color_by == 'asu_id':
            idx = asu_id
            ndx = self.crystal.model.n_asu
        elif self.color_by == 'unit_cell':
            idx = self.crystal.hkl_to_id(unit_cell)
            ndx = self.crystal.n_cell
        if self.color_palette == 'xkcd':
            color_dict = mcolors.XKCD_COLORS
        elif self.color_palette == 'tableau':
            color_dict = mcolors.TABLEAU_COLORS
        else:
            color_dict = mcolors.CSS4_COLORS
        color_array = np.array(list(color_dict.items()))
        return color_array[::color_array.shape[0]//ndx,1][idx]

class PhononPlots:

    def __init__(self, phonon):
        self.phonon = phonon

    def _get_dispersion(self, h=True, k=True, l=True):
        w = np.sqrt(1. / np.real(self.phonon.Winv))
        k_norm = np.zeros((self.phonon.hsampling[2]))
        w_curve = np.zeros((self.phonon.hsampling[2], w.shape[-1]))
        for i in range(self.phonon.hsampling[2]):
            w_curve[i] = w[h * i, k * i, l * i]
            k_norm[i] = self.phonon.kvec_norm[h * i, k * i, l * i]
        return k_norm, w_curve

    def dispersion_curve(self):
        nrows = 2
        ncols = 4
        fig = plt.figure(figsize=(2 * ncols, 4 * nrows), dpi=180,
                         constrained_layout=True)
        gs = GridSpec(nrows, ncols, figure=fig)

        title   = ['0->h','0->k','0->l','0->h+k','0->h+l','0->k+l','0->h+k+l']
        h_curve = [True,  False, False, True,  True,  False, True]
        k_curve = [False, True,  False, True,  False, True,  True]
        l_curve = [False, False, True,  False, True,  True,  True]

        for i_curve in range(8):
            gs_j = i_curve % ncols
            gs_i = i_curve // ncols
            ax = fig.add_subplot(gs[gs_i, gs_j])
            if i_curve == 0:
                ax_save = ax
            if i_curve < 7:
                ax.sharex(ax_save)
                ax.sharey(ax_save)
                knorm, wvec = self._get_dispersion(h=h_curve[i_curve],
                                                   k=k_curve[i_curve],
                                                   l=l_curve[i_curve])
                for i in range(wvec.shape[-1]):
                    ax.plot(knorm, wvec[:, i], 'o', label=f'#{i}')
                    if gs_i == 1:
                        ax.set_xlabel('phonon wavevector ($\mathrm{\AA}^{-1}$)')
                    if gs_j == 0:
                        ax.set_ylabel('phonon frequency')
                if i_curve == 3:
                    ax.legend(bbox_to_anchor=(1.1,0.5))
                ax.set_title(title[i_curve])
            else:
                ax.hist(np.sqrt(1. / np.real(self.phonon.Winv).flatten()),
                        bins=50, orientation='horizontal')
                ax.set_title('density of states')
        plt.tight_layout()
        plt.show()

    def contribution_curve(self):
        nrows=2
        ncols=3
        fig = plt.figure(figsize=(2 * ncols, 3 * nrows), dpi=180,
                         constrained_layout=True)
        gs = GridSpec(nrows, ncols, figure=fig)
        knorm = self.phonon.kvec_norm.reshape(-1,1)
        Winv2 = np.real(self.phonon.Winv).reshape(-1,6)
        Vvec  = self.phonon.V.reshape(-1,6,6)
        A = ['x', 'y', 'z', 'r1', 'r2', 'r3']

        for i_curve in range(6):
            gs_j = i_curve % ncols
            gs_i = i_curve // ncols
            ax = fig.add_subplot(gs[gs_i, gs_j])
            if i_curve == 0:
                ax_save = ax
            ax.sharex(ax_save)
            ax.sharey(ax_save)
            for i in range(6):
                ax.plot(knorm,
                        Winv2[:,i_curve]*np.abs(Vvec[:,i,i_curve]),
                        '.', label=f'{A[i]}')
            ax.set_title(f'#{i_curve}')
            if gs_i == 1:
                ax.set_xlabel('phonon wavevector ($\mathrm{\AA}^{-1}$)')
            if gs_j == 0:
                ax.set_ylabel('phonon intensity')
            ax.set_yscale('log')
            if i_curve == 2:
                ax.legend(bbox_to_anchor=(2.1,0.5))
        plt.show()
