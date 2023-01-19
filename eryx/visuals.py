import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go

def visualize_central_slices(I, vmax_scale=5):
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
    
    ax1.imshow(I[int(map_shape[0]/2),:,:], vmax=vmax)
    ax2.imshow(I[:,int(map_shape[1]/2),:], vmax=vmax)
    ax3.imshow(I[:,:,int(map_shape[2]/2)], vmax=vmax)

    ax1.set_aspect(map_shape[2]/map_shape[1])
    ax2.set_aspect(map_shape[2]/map_shape[0])
    ax3.set_aspect(map_shape[1]/map_shape[0])

    ax1.set_title("(0,k,l)", fontsize=14)
    ax2.set_title("(h,0,l)", fontsize=14)
    ax3.set_title("(h,k,0)", fontsize=14)

    for ax in [ax1,ax2,ax3]:
        ax.set_xticks([])
        ax.set_yticks([])


class VisualizeCrystal:

    def __init__(self, crystal):
        """
        Plotly helper functions to visualize the Crystal object.
        Parameters
        ----------
        crystal : eryx.models.Crystal object
        """
        self.crystal = crystal
        self.draw_data = None
        self.color_by = 'asu_id'
        self.color_palette = 'xkcd'

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
                                      name=f'Cell #{self.crystal._hkl_to_id(unit_cell=[h,k,l])} | ASU #{i_asu}')

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
            idx = self.crystal._hkl_to_id(unit_cell)
            ndx = self.crystal.n_cell
        if self.color_palette == 'xkcd':
            color_dict = mcolors.XKCD_COLORS
        elif self.color_palette == 'tableau':
            color_dict = mcolors.TABLEAU_COLORS
        else:
            color_dict = mcolors.CSS4_COLORS
        color_array = np.array(list(color_dict.items()))
        return color_array[::color_array.shape[0]//ndx,1][idx]
