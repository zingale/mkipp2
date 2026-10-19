#!/usr/bin/env python
"""Demonstrate Kippenhahn plots and custom data extraction from MESA output."""

import mkipp2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch


def plot1():
    # simple example (saves to default filename Kippenhahn.png)

    k = mkipp2.KippenhahnDiagram()
    k.plot()


def plot2():
    # specify xrange and filename

    k = mkipp2.KippenhahnDiagram(xlims=[300, 600])
    k.plot(save_filename="Kippenhahn2.png")


def plot3():
    # plot of Helium abundance against time, independent decoration

    fig, axis = plt.subplots()

    # mkipp2.KippenhahnDiagram returns an object containing
    #   .contour_plot : the return value of matplotlibs contourf. Can be
    #                            used to create a colorbar with plt.colorbar()
    #   .histories    : list of history files read. data can be accessed
    #                            from this using the get("column_name") function
    #   xlims        : limits of data in x coordinate
    k = mkipp2.KippenhahnDiagram(xaxis="star_age",
                                 time_units="Myr",
                                 identifier="y",
                                 log10_on_data=False)
    plot = k.plot(levels=np.arange(0.0, 1.001, 0.01),
                  axis=axis, decorate_plot=False, save_file=False)
    cbar = plt.colorbar(plot.contour_plot, pad=0.05)
    cbar.set_label("Helium abundance")
    axis.set_xlabel("Time (Myr)")
    axis.set_ylabel("Mass (solar masses)")
    axis.set_xlim(plot.xlims)
    fig.tight_layout()
    fig.savefig("Kippenhahn3.png")


def plot4():
    # read out max age of star first, then create a log(tf-t) plot

    fig, axis = plt.subplots()

    # only need to read star_age column first
    history = mkipp2.mesa_data.MesaData("LOGS/history.data",
                                        read_data_cols=["star_age"])
    max_age = max(history.get("star_age"))
    k = mkipp2.KippenhahnDiagram(xaxis="star_age",
                                 time_units="yr",
                                 function_on_xaxis=lambda x: np.log10(max_age+0.01 - x))
    kipp_plot = k.plot(decorate_plot=False, save_file=False, axis=axis)
    cbar = plt.colorbar(kipp_plot.contour_plot, pad=0.05)
    cbar.set_label("log |eps_nuc|")
    axis.set_xlabel("log (tf-t) [yrs]")
    axis.set_ylabel("Mass (solar masses)")
    fig.tight_layout()
    fig.savefig("Kippenhahn4.png")


def plot5():
    # add custom extractor to compute g
    fig, axis = plt.subplots()
    msun = 1.99e33
    rsun = 6.96e10
    cgrav = 6.67e-8

    def g_extractor(identifier, log10_on_data, prof, return_data_columns=False):
        """Return gravitational acceleration, its logarithm, or required columns."""
        if return_data_columns:
            return ["radius", "mass"]
        data = cgrav*prof.get("mass")*msun/(prof.get("radius")*rsun)**2
        if log10_on_data:
            return np.log10(data)
        return data

    k = mkipp2.KippenhahnDiagram(extractor=g_extractor)
    plot = k.plot(decorate_plot=False, save_file=False, axis=axis)
    cbar = plt.colorbar(plot.contour_plot, pad=0.05)
    cbar.set_label("log g")
    axis.set_xlabel("model_number")
    axis.set_ylabel("Mass (solar masses)")
    fig.tight_layout()
    fig.savefig("Kippenhahn5.png")


def plot6():
    # Reading out mixing regions and data, and plotting independently

    k = mkipp2.KippenhahnDiagram()

    fig, axis = plt.subplots()

    # reuse profile data prepared by the constructor
    # k.xyz_data contains
    #   xyz_data.xlims : limits of data in x coordinate
    #   xyz_data.X     : 2D array of horizontal coordinates
    #   xyz_data.Y     : 2D array of vertical coordinates (mass or radius)
    #   xyz_data.Z     : 2D array of interpolated values from the extractor
    # the last three can be used as inputs for matplotlib contour or contourf
    xyz_data = k.xyz_data

    # reuse mixing regions prepared by the constructor
    # k.mixing_zones contains
    #   mixing_zones.zones     : matplotlib Path objects for each mixing zone.
    #                            These can be plotted using add_patch(...)
    #   mixing_zones.mix_types : Integer array containing the type of each zone
    #   mixing_zones.x_coords  : x coordinates for points at the surface
    #   mixing_zones.y_coords  : y coordinates for points at the surface
    #   mixing_zones.histories : mesa_data history files to access additional data
    mixing_zones = k.mixing_zones

    # just plot convection, overshooting and semiconvection
    for i, zone in enumerate(mixing_zones.zones):
        color = ""
        # Convective mixing
        if mixing_zones.mix_types[i] == 1:  # convection
            color = '#332288'
        # Overshooting
        elif mixing_zones.mix_types[i] == 3:  # overshooting
            color = '#117733'
        # Semiconvective mixing
        elif mixing_zones.mix_types[i] == 4:  # semiconvection
            color = '#CC6677'
        else:
            continue
        axis.add_patch(PathPatch(zone, color=color, alpha=0.5, lw=0))

    if (xyz_data.Z.size > 0):
        cs = plt.contour(xyz_data.X, xyz_data.Y, xyz_data.Z, [0, 4, 8], colors='k')
        plt.clabel(cs, inline=1, fontsize=10)

    axis.plot(mixing_zones.x_coords, mixing_zones.y_coords, 'k', lw=4)
    axis.set_xlabel("model_number")
    axis.set_ylabel("Mass (solar masses)")
    axis.set_xlim(0, max(mixing_zones.x_coords))
    axis.set_ylim(0, max(mixing_zones.y_coords))
    fig.tight_layout()
    fig.savefig("Kippenhahn6.png")


def plot7():
    # create a multi-panel plot

    fig = plt.figure(figsize=(9, 5), layout="constrained")

    spec = fig.add_gridspec(ncols=4, nrows=1, wspace=0.05, width_ratios=[5, 4, 4, 0.75])
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])
    ax4 = fig.add_subplot(spec[0, 3])

    k = mkipp2.KippenhahnDiagram(xaxis="star_age")
    k.plot(axis=ax1, num_levels=10,
           save_file=False, decorate_plot=False)
    k.plot(axis=ax2, num_levels=10,
           save_file=False, decorate_plot=False,
           show_mixing_legend=False, show_core_mass_legend=False)
    plot = k.plot(axis=ax3, num_levels=10,
                  save_file=False, decorate_plot=False,
                  show_mixing_legend=False, show_core_mass_legend=False)
    cbar = fig.colorbar(plot.contour_plot, cax=ax4)
    #cbar.ax.set_yticklabels(['0', '','2','','4','','6','','8',''])
    cbar.set_label(r"$\log_{10}|\epsilon_{\rm nuc}|$")

    history = mkipp2.mesa_data.MesaData("LOGS/history.data",
                                        read_data_cols=["star_age", "center_h1", "center_he4"])
    for i, center_h1 in enumerate(history.get("center_h1")):
        if center_h1 < 1e-3:
            age_tams = history.get("star_age")[i] / 1e6
            break
    for i, center_he4 in enumerate(history.get("center_he4")):
        if center_he4 < 1e-3:
            age_hedep = history.get("star_age")[i] / 1e6
            break

    ax1.set_xlim([0, age_tams])
    ax2.set_xlim([age_tams, age_hedep])
    ax3.set_xlim([age_hedep, history.get("star_age")[-1] / 1e6])
    ax2.set_xlabel("star age (Myr)")
    ax1.set_ylabel("Mass (solar masses)")
    ax2.yaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    fig.savefig("Kippenhahn7.png")


if __name__ == "__main__":
    plot1()
    plot2()
    plot3()
    plot4()
    plot5()
    plot6()
    plot7()
