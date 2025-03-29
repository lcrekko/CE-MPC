"""
plotter.py

This module contains
(1) The basic plotting class that do statistical plots for a table format data,
frequently used in Monte Carlo simulation
(2) it also has a zoom-in submodule to plot a specific zoomed-in plot on top of the original plot
(3) Additional functions that will be used in the class

The plotting uses the basic matplotlib package
"""

import bisect
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

def find_closest_index(a_vec, b):
    """
    Finds the index of a value in a_vec that is closed to b
    :param a_vec: the array a_vec
    :param b: the value b
    :return: the index
    Remark: this function is used for zoom-in plots
    """
    # Handle edge cases
    if not a_vec.all():
        return None
    if b <= a_vec[0]:
        return 0
    if b >= a_vec[-1]:
        return len(a_vec) - 1

    # Find the insertion point for b in the sorted array a
    index = bisect.bisect_left(a_vec, b)

    # Determine the closest index
    if index == 0:
        return 0
    if index == len(a_vec):
        return len(a_vec) - 1
    return min(index, index - 1, key=lambda i: abs(a_vec[i] - b))


def plotter_kernel(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                    info_text: dict, info_color: tuple,
                    marker=False) -> None:
    """
    This is the kernel of the main plotting function MonteCarloPlotter
    :param ax: the handle of the subplots, e.g., ax[1, 0] (the second row, first column)
    :param x_data: the x-axis data (1-D array)
    :param y_data: a bunch of y-axis data (2-D array)
    :param info_text: the text information, a dictionary that contains labels and titles
                1) "x_label": the text of the x-axis
                2) "y_label": the text of the y-axis
                3) "legend": legend information, may not be used at all
    :param info_color: tuple, color information, just a color
    :param marker: whether to show the marker
    Remark: this function will be frequently used in the class MonteCarloPlotter()
    """
    # Compute the max, min, mena and variance
    y_max = np.max(y_data, axis=0)
    y_min = np.min(y_data, axis=0)
    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0)

    # Creat color variations
    color_bound = tuple(x * 0.75 for x in info_color)
    color_variance = tuple(x * 0.5 for x in info_color)
    color_range = tuple(x * 0.25 for x in info_color)

    # basic mean plot
    if marker:
        color_marker_face = (info_color[0] * 0.75, np.min([1, info_color[1] * 1.25]), info_color[2])
        color_marker_edge = (np.min([1, info_color[0] * 1.25]), info_color[1] * 0.75, info_color[2])
        ax.plot(x_data, y_mean,
                label=info_text["legend"],
                linewidth=2.5, color=info_color,
                marker='.', markersize=10, markerfacecolor=color_marker_face,
                markeredgewidth=2, markeredgecolor=color_marker_edge)
    else:
        ax.plot(x_data, y_mean,
                label=info_text["legend"],
                linewidth=2.5, color=info_color)

    # plot the max and min
    ax.plot(x_data, y_min,
            linewidth=1.5, linestyle=':', color=color_bound)
    ax.plot(x_data, y_max,
            linewidth=1.5, linestyle=':', color=color_bound)

    # plot the upper variance and lower variance
    ax.plot(x_data, y_mean - y_std,
            linewidth=1.5, linestyle='--', color=color_bound)
    ax.plot(x_data, y_mean + y_std,
            linewidth=1.5, linestyle='--', color=color_bound)

    # plot the variance (fill the shaded color)
    ax.fill_between(x_data, y_mean - y_std, y_mean + y_std, color=color_variance, alpha=0.25)

    # plot the max and the min (fill the shaded color)
    ax.fill_between(x_data, y_min, y_max, color=color_range, alpha=0.125)


def plotter_kernel_multi_traj(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                              info_color=(0.15, 0.15, 0.15),
                              marker=False) -> None:
    """
    This is the kernel of the main plotting function MonteCarloPlotter
    :param ax: the handle of the subplots, e.g., ax[1, 0] (the second row, first column)
    :param x_data: the x-axis data (1-D array)
    :param y_data: a bunch of y-axis data (2-D array)
    :param info_text: the text information, a dictionary that contains labels and titles
                1) "x_label": the text of the x-axis
                2) "y_label": the text of the y-axis
                3) "legend": legend information, may not be used at all
    :param info_color: tuple, color information, just a color
    :param marker: whether to show the marker
    Remark: this function will be used in the class MonteCarloPlotter()
    """
    # compute the min and max to plot the envolope
    y_max = np.max(y_data, axis=0)
    y_min = np.min(y_data, axis=0)

    # set the color to fill in the boundary
    color_bound = tuple(x * 0.75 for x in info_color)
    color_range = tuple(x * 0.05 for x in info_color)

    # looping to plot every trajectory
    if marker:
        for i in range(y_data.shape[0]):
            ax.plot(x_data, y_data[i,:], linewidth=2.5,
                    marker='.', markersize=10, markeredgewidth=2)
    else:
        for i in range(y_data.shape[0]):
            ax.plot(x_data, y_data[i,:], linewidth=2.5)

    # plot the max and min
    ax.plot(x_data, y_min,
            linewidth=1.5, linestyle=':', color=color_bound)
    ax.plot(x_data, y_max,
            linewidth=1.5, linestyle=':', color=color_bound)

    # fill in the min and max
    ax.fill_between(x_data, y_min, y_max, color=color_range, alpha=0.125)


class MonteCarloPlotter():
    """
    This is the Monte Carlo Plotter class, it has two parts
    1. Initialization using the figure handle, data, text, and color information
    2. Plot the mean-variance-bound plot
    3. Plot the mean-variance-bound plot with zoomed-in
    """
    def __init__(self, ax: plt.Axes,
                 x_data: np.ndarray, y_data:np.ndarray,
                 info_text: dict, info_font: dict, info_color: tuple,
                 marker=False):
        """
        Initialize the plotter

        Parameters:
            ax: the handle of the subplots or a plot
            x_data: the x-axis data, 1-D array
            y_data: the y-axis data, 2-D array
            info_text: the text info of the plot, a dictionary, with the following information
                1) "x_label": the text of the x-axis
                2) "y_label": the text of the y-axis
                3) "legend": legend information, may not be used at all
            info_font: the font info of the plot, a dictionary, with the following information
                1) "ft_type": the type of the font
                2) "ft_size_label": label size
                3) "ft_size_legend": legend size
                4) "ft_size_tick": tick size
            info_color: a tuple, the theme color of the plot
            marker: BOOL, no marker is added by default
        """

        # Pass the information
        self.ax = ax
        self.x_data = x_data
        self.y_data = y_data
        self.info_text = info_text
        self.info_font = info_font
        self.info_color = info_color
        self.marker = marker


    def plot_basic(self, x_scale_log=False, y_scale_log=False, set_x_ticks=False):
        """
        This function is the basic plot, without any zoom in.

        Parameter:
            x_scale_log: whether the x_scale use the log
            y_scale_log: whether the y_scale use the log
            set_x_ticks: whether we actively control the ticks
        """
        # ----------- Plotting Section -----------
        plotter_kernel(self.ax,
                       self.x_data, self.y_data,
                       self.info_text, self.info_color,
                       self.marker)

        # ----------- Post-configuration -----------
        # set the title and the labels
        self.ax.set_xlabel(self.info_text["x_label"],
                            fontdict={'family': self.info_font["ft_type"],
                                      'size': self.info_font["ft_size_label"],
                                      'weight': 'bold'})
        self.ax.set_ylabel(self.info_text["y_label"],
                            fontdict={'family': self.info_font["ft_type"],
                                      'size': self.info_font["ft_size_label"],
                                      'weight': 'bold'})
        # set the x-axis
        if set_x_ticks:
            self.ax.set_xticks(self.x_data)
            # # Manually set tick labels as 1, 2, ..., 10
            # self.ax.set_xticklabels([str(i) for i in range(1, 11)])

            # # Hide default offset text
            # self.ax.xaxis.get_offset_text().set_visible(False)

            #     # Add the "× 10⁻⁴" manually below the axis
            # self.ax.annotate(r"$\times 10^{-3}$",
            #         xy=(1, -0.05),
            #         xycoords='axes fraction',
            #         ha='center', fontsize=20)
        # ax.set_ylabel('Y Label')

        # Use ScalarFormatter and disable offset notation
        formatter = mticker.ScalarFormatter(useOffset=False)
        self.ax.yaxis.set_major_formatter(formatter)

        # set size of the ticks
        self.ax.tick_params(axis='x', labelsize=self.info_font["ft_size_tick"])
        self.ax.tick_params(axis='y', labelsize=self.info_font["ft_size_tick"])

        # set the legend
        dummy_line = mlines.Line2D([], [], color='none')
        self.ax.legend([dummy_line], [self.info_text["legend"]],
                       loc='upper left', fontsize=self.info_font["ft_size_legend"],
                        prop={'family': self.info_font["ft_type"],
                              'size': self.info_font["ft_size_legend"]}, bbox_to_anchor=(-0.1, 1),
                              frameon=False)

        # set the log-scale
        if x_scale_log:
            self.ax.set_xscale('log')
        if y_scale_log:
            self.ax.set_yscale('log')

        # set the background and grid
        self.ax.set_facecolor((0.95, 0.95, 0.95))
        self.ax.grid(True, linestyle='--', color='white', linewidth=1)

    def plot_basic_multi_traj(self, x_scale_log=False, y_scale_log=False, set_x_ticks=False):
        """
        This function is the basic plot, without any zoom in.

        Parameter:
            x_scale_log: whether the x_scale use the log
            y_scale_log: whether the y_scale use the log
            set_x_ticks: whether we actively control the ticks
        """
        # ----------- Plotting Section -----------
        plotter_kernel_multi_traj(self.ax,
                                  self.x_data, self.y_data,
                                  self.info_color,
                                  self.marker)

        # ----------- Post-configuration -----------
        # set the title and the labels
        self.ax.set_xlabel(self.info_text["x_label"],
                            fontdict={'family': self.info_font["ft_type"],
                                      'size': self.info_font["ft_size_label"],
                                      'weight': 'bold'})
        # set the x-axis
        if set_x_ticks:
            self.ax.set_xticks(self.x_data)
            # # Manually set tick labels as 1, 2, ..., 10
            # self.ax.set_xticklabels([str(i) for i in range(1, 11)])

            # # Hide default offset text
            # self.ax.xaxis.get_offset_text().set_visible(False)

            #     # Add the "× 10⁻⁴" manually below the axis
            # self.ax.annotate(r"$\times 10^{-3}$",
            #         xy=(1, -0.05),
            #         xycoords='axes fraction',
            #         ha='center', fontsize=20)
        # ax.set_ylabel('Y Label')

        formatter = mticker.ScalarFormatter(useOffset=False)
        self.ax.yaxis.set_major_formatter(formatter)

        # set size of the ticks
        self.ax.tick_params(axis='x', labelsize=self.info_font["ft_size_tick"])
        self.ax.tick_params(axis='y', labelsize=self.info_font["ft_size_tick"])

        # set the legend
        dummy_line = mlines.Line2D([], [], color='none')
        self.ax.legend([dummy_line], [self.info_text["legend"]],
                       loc='upper left', fontsize=self.info_font["ft_size_legend"],
                        prop={'family': self.info_font["ft_type"],
                              'size': self.info_font["ft_size_legend"]}, bbox_to_anchor=(-0.2, 1.05),
                              frameon=False)

        # set the log-scale
        if x_scale_log:
            self.ax.set_xscale('log')
        if y_scale_log:
            self.ax.set_yscale('log')

        # set the background and grid
        self.ax.set_facecolor((0.95, 0.95, 0.95))
        self.ax.grid(True, linestyle='--', color='white', linewidth=1)

    def plot_zoom(self, info_zoom, x_scale_log=False, y_scale_log=False, set_x_ticks=False):
        """
        This function is the basic plot, with the zoom in module.

        Parameter:
            info_zoom: dict, the zoom-in info, the details are
                      1. 'zoom' (boolean) True or False
                      2. 'ratio' (float) ratio of the zoomed in
                      3. 'loc' (str) location
                      4. 'x_range' (tuple) (x_min, x_max)
                      5. 'set_x_ticks' (bool) whether to set the x_ticks
                      6. 'x_ticks' (list) the x ticks
                      7. 'y_auto' (boolean) True of False, if it is True, the automatic range will be
                      adjusted accordingly based on the x range
                      8. 'y_range' (tuple) (y_min, y_max)
            x_scale_log: whether the x_scale use the log
            y_scale_log: whether the y_scale use the log
            set_x_ticks: whether we actively control the ticks
        """
        # ----------- Basic Plotting Section -----------
        plotter_kernel(self.ax,
                       self.x_data, self.y_data,
                        self.info_text, self.info_color,
                        self.marker)

        # ----------- Zoom-in Plotting ------------
        ax_zoom = zoomed_inset_axes(self.ax, zoom=info_zoom['ratio'], loc=info_zoom['loc'])
        plotter_kernel(ax_zoom, self.x_data, self.y_data,
                        self.info_text, self.info_color,
                        self.marker)
        # Set the limits of the inset axes to zoom in on a specific area
        ax_zoom.set_xlim(info_zoom['x_range'][0], info_zoom['x_range'][1])
        if info_zoom['set_x_ticks']:
            ax_zoom.set_xticks(info_zoom['x_ticks'])
        ax_zoom.xaxis.tick_top()
        if info_zoom['y_auto']:
            zoom_y_max = 1.001 * np.max(self.y_data, axis=0)
            zoom_y_min = 0.999 * np.min(self.y_data, axis=0)
            zoom_x_left = find_closest_index(self.x_data, info_zoom['x_range'][0])
            zoom_x_right = find_closest_index(self.x_data, info_zoom['x_range'][1])
            ax_zoom.set_ylim(np.min([zoom_y_min[zoom_x_left], zoom_y_min[zoom_x_right]]),
                             np.max([zoom_y_max[zoom_x_left], zoom_y_max[zoom_x_right]]))
        else:
            ax_zoom.set_ylim(info_zoom['y_range'][0], info_zoom['y_range'][1])

        # Mark the region of interest on the main plot
        mark_inset(self.ax, ax_zoom, loc1=1, loc2=3, fc="none", ec="black")
        ax_zoom.tick_params(axis='x', labelsize=0.5 * self.info_font["ft_size_tick"])
        ax_zoom.tick_params(axis='y', labelsize=0.5 * self.info_font["ft_size_tick"])

        # ----------- Post-configuration -----------
        # set the title and the labels
        # self.ax.set_title(info_text['title'], fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        self.ax.set_xlabel(self.info_text["x_label"],
                            fontdict={'family': self.info_font["ft_type"],
                                      'size': self.info_font["ft_size_label"],
                                      'weight': 'bold'})
        # set the x-axis
        if set_x_ticks:
            self.ax.set_xticks(self.x_data)
        # ax.set_ylabel('Y Label')

        # set size of the ticks
        self.ax.tick_params(axis='x', labelsize=self.info_font["ft_size_tick"])
        self.ax.tick_params(axis='y', labelsize=self.info_font["ft_size_tick"])

        # set the legend
        self.ax.legend(loc='upper left', fontsize=self.info_font["ft_size_legend"],
                        prop={'family': self.info_font["ft_type"],
                              'size': self.info_font["ft_size_legend"]})

        # set the log-scale
        if x_scale_log:
            self.ax.set_xscale('log')
        if y_scale_log:
            self.ax.set_yscale('log')

        # set the background and grid
        self.ax.set_facecolor((0.95, 0.95, 0.95))
        self.ax.grid(True, linestyle='--', color='white', linewidth=1)

class ManifoldPlotter():
    """
    This class will plot 3D scatter points for intended function f(x_1,x_2), where the data-points
    can be generated according to some specified level set function r = g(x_1,x_2), and fitting is also
    enabled to add a surface which shows the interpolated behavior.

    It will first initialize using data and then do the plotting with plotter specification
    """
    def __init__(self, r_values, x1_values, func_level_set, func_y, x2_max):
        """
        This initialization function will generate the desired data for plotting

        Parameters:
            r_values: ndarray, the level set value
            x1_values: the baseline x_values
            func_level_set: the level set function x_2 = g^{-1}(x_1;r)
            func_y: the z value function
            x2_max: the maximum allowed value of x2
        """
        # Generate data points along level sets
        x1_vals, x2_vals, y_vals = [], [], []
        x1_max = np.max(x1_values)
        x1_min = np.min(x1_values)

        for r in r_values:
            x2_values = func_level_set(x1_values, r)
            valid_idx = np.where(x2_values <= x2_max)[0]  # Indices where x2 is valid

            for i in valid_idx:  # Loop over valid indices to apply user-defined function
                x1_vals.append(x1_values[i])
                x2_vals.append(x2_values[i])
                y_vals.append(func_y(x1_values[i], x2_values[i]))  # Call the function on scalars

        # Convert lists to numpy arrays
        self.x1_scatter = np.array(x1_vals)
        self.x2_scatter = np.array(x2_vals)
        self.y_scatter = np.array(y_vals)

        # Create interpolation grid
        x1_grid = np.linspace(0.1 * x1_min, x1_max, 100)
        x2_grid = np.linspace(1e-3, x2_max, 100)
        self.x1_grid_mesh, self.x2_grid_mesh = np.meshgrid(x1_grid, x2_grid)

        # Performe the interpolation
        self.y_grid_mesh = griddata((self.x1_scatter, self.x2_scatter), self.y_scatter,
                                    (self.x1_grid_mesh, self.x2_grid_mesh), method='cubic')

    def plotter_basic(self, ax, info_text, info_font):
        """
        The plotter function that plot the data based on the given
        plotter handle and text information

        Parameters:
            ax: plotter handle
            info_text: the text info of the plot, a dictionary, with the following information
                1) "x_label": the text of the x-axis
                2) "y_label": the text of the y-axis
                3) "legend": legend information, may not be used at all
            info_font: the font info of the plot, a dictionary, with the following information
                1) "ft_type": the type of the font
                2) "ft_size_label": label size
                3) "ft_size_legend": legend size
                4) "ft_size_tick": tick size
        """
        # Scatter plot of original points
        ax.scatter(self.x1_scatter, self.x2_scatter, self.y_scatter,
                   c=self.y_scatter, cmap='viridis', s=20, edgecolor='k', label="Sample Points")

        # Plot interpolated surface
        ax.plot_surface(self.x1_grid_mesh, self.x2_grid_mesh, self.y_grid_mesh,
                        cmap='viridis', alpha=0.6, edgecolor='none')

        # Labels
        ax.set_xlabel(info_text["x_label"], labelpad=15,
                      fontdict={'family': info_font["ft_type"],
                                      'size': info_font["ft_size_label"],
                                      'weight': 'bold'})
        ax.set_ylabel(info_text["y_label"], labelpad=15,
                      fontdict={'family': info_font["ft_type"],
                                      'size': info_font["ft_size_label"],
                                      'weight': 'bold'})
        ax.set_zlabel(info_text["legend"], labelpad=20,
                      fontdict={'family': info_font["ft_type"],
                                      'size': info_font["ft_size_label"],
                                      'weight': 'bold'})
        # set size of the ticks
        ax.tick_params(axis='x', labelsize=info_font["ft_size_tick"], pad=1)
        ax.tick_params(axis='y', labelsize=info_font["ft_size_tick"], pad=1)
        ax.tick_params(axis='z', labelsize=info_font["ft_size_tick"], pad=10)
