import numpy as np
import matplotlib.pyplot as plt

import mkipp2


def basic_plot():

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 identifier="net_nuclear_energy",
                                 log10_on_data=False)
    k.plot(core_masses=["He", "CO", "ONe", "Fe"],
           size=(8, 6),
           save_filename="Kipp-basic.png")


def age_plot():

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="star_age",
                                 time_units="yr",
                                 identifier="net_nuclear_energy",
                                 log10_on_data=False)
    k.plot(core_masses=["He", "CO", "ONe", "Fe"],
           size=(8, 6),
           save_filename="Kipp-age.png")


def collapse_plot():

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="collapse",
                                 time_units="yr",
                                 identifier="net_nuclear_energy",
                                 log10_on_data=False)

    k.plot(core_masses=["He", "CO", "ONe", "Fe"],
           size=(8, 6),
           save_filename="Kipp-collapse.png")


def density_contours_plot():

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="collapse",
                                 time_units="yr",
                                 identifier="net_nuclear_energy",
                                 log10_on_data=False)

    k.plot(core_masses=["He", "CO", "ONe", "Fe"],
           density_contours=[1.e2, 1.e4, 1.e6, 1.e8],
           size=(8, 6),
           save_filename="Kipp-collapse-density.png")


def si_plot():

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="collapse",
                                 time_units="yr",
                                 identifier="si28",
                                 log10_on_data=True)

    k.plot(core_masses=["He", "CO", "ONe", "Fe"],
           density_contours=[1.e2, 1.e4, 1.e6, 1.e8],
           size=(8, 6),
           levels=np.arange(-6, 0.5, 0.5),
           save_filename="Kipp-si.png")


def burning_regions_plot():

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="collapse",
                                 time_units="yr",
                                 load_profiles=False)

    k.plot_burning_regions(core_masses=["He", "CO", "ONe", "Fe"],
                           size=(8, 6),
                           save_filename="Kipp-burning_regions.png")


def multipanel():
    # create a multi-panel plot

    fig = plt.figure(figsize=(11, 5), layout="constrained")

    spec = fig.add_gridspec(ncols=5, nrows=1, wspace=0.05,
                            width_ratios=[6, 4, 4, 4, 0.75])
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])
    ax4 = fig.add_subplot(spec[0, 3])
    ax5 = fig.add_subplot(spec[0, 4])

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 identifier="net_nuclear_energy",
                                 xaxis="star_age")

    k.plot(axis=ax1, num_levels=10,
           save_file=False, decorate_plot=False)
    k.plot(axis=ax2, num_levels=10,
           save_file=False, decorate_plot=False,
           show_mixing_legend=False, show_core_mass_legend=False)
    k.plot(axis=ax3, num_levels=10,
           save_file=False, decorate_plot=False,
           show_mixing_legend=False, show_core_mass_legend=False)
    plot = k.plot(axis=ax4, num_levels=10,
                  save_file=False, decorate_plot=False,
                  show_mixing_legend=False, show_core_mass_legend=False)

    cbar = fig.colorbar(plot.contour_plot, cax=ax5)
    #cbar.ax.set_yticklabels(['0', '','2','','4','','6','','8',''])
    cbar.set_label(r"$\log_{10}|\epsilon_{\rm nuc}|$")

    history = mkipp2.mesa_data.MesaData("LOGS-mesa-26.04.1.zip/history.data",
                                        read_data_cols=["star_age", "center_h1", "center_he4", "center_o16"])
    for i, center_h1 in enumerate(history.get("center_h1")):
        if center_h1 < 1e-3:
            age_tams = history.get("star_age")[i] / 1e6
            break
    for i, center_he4 in enumerate(history.get("center_he4")):
        if center_he4 < 1e-3:
            age_hedep = history.get("star_age")[i] / 1e6
            ihedep = i
            break
    for i, center_o16 in enumerate(history.get("center_o16")):
        if i < ihedep:
            continue
        if center_o16 < 1e-3:
            age_odep = history.get("star_age")[i] / 1e6
            break

    ax1.set_xlim([0, age_tams])
    ax2.set_xlim([age_tams, age_hedep])
    ax3.set_xlim([age_hedep, age_odep])
    ax4.set_xlim([age_odep, history.get("star_age")[-1] / 1e6])

    ax2.set_xlabel("star age (Myr)")
    ax1.set_ylabel("Mass (solar masses)")

    ax2.yaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    fig.savefig("Kipp-split.png")


if __name__ == "__main__":

    #basic_plot()
    #age_plot()
    #collapse_plot()
    #density_contours_plot()
    #si_plot()
    #burning_regions_plot()
    multipanel()
