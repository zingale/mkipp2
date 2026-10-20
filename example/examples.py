import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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


def hybrid_burning_regions_plot():

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="collapse",
                                 time_units="yr",
                                 identifier="net_nuclear_energy")

    k.plot_burning_regions(hybrid_mode=True,
                           core_masses=["He", "CO", "ONe", "Fe"],
                           size=(8, 6),
                           save_filename="Kipp-burning_regions-hybrid.png")


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
    cbar.set_label(r"$\log_{10}|\epsilon_{\rm net}|$")

    # we want to set the widths of each panel according to
    # the exhaustion of different fuels

    history = mkipp2.mesa_data.MesaData("LOGS-mesa-26.04.1.zip/history.data",
                                        read_data_cols=["star_age", "center_h1",
                                                        "center_he4", "center_o16"])

    nuclei = ["h1", "he4", "o16"]
    bounds = {}
    ilast = -1
    for inuc, nuc in enumerate(nuclei):
        for i, center_nuc in enumerate(history.get(f"center_{nuc}")):
            if i < ilast:
                continue
            if center_nuc < 1e-3:
                bounds[nuc] = history.get("star_age")[i] / 1e6
                ilast = i
                break

    ax1.set_xlim([0, bounds["h1"]])
    ax2.set_xlim([bounds["h1"], bounds["he4"]])
    ax3.set_xlim([bounds["he4"], bounds["o16"]])
    ax4.set_xlim([bounds["o16"], history.get("star_age")[-1] / 1e6])

    for ax in (ax1, ax2, ax3, ax4):
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.tick_params(axis="both", labelsize=9)

    # for the last panel, the timescale is so short that we need
    # to finesse the tick labels.
    ax4.xaxis.set_major_locator(LinearLocator(2))
    ax4.xaxis.set_major_formatter(FormatStrFormatter("%.9f"))

    # Keep the long endpoint labels inside ax4 so they don't widen the gaps.
    ax4_labels = ax4.get_xticklabels()
    ax4_labels[0].set_horizontalalignment("left")
    ax4_labels[-1].set_horizontalalignment("right")

    ax1.set_ylabel("Mass (solar masses)")
    fig.supxlabel("star age (Myr)", x=0.5, ha="center", va="bottom",
                  fontsize=ax1.yaxis.label.get_fontsize())

    for ax in (ax2, ax3, ax4):
        ax.tick_params(axis="y", left=False, right=False,
                       labelleft=False, labelright=False, which="both")
    fig.savefig("Kipp-split.png")


if __name__ == "__main__":

    basic_plot()
    age_plot()
    collapse_plot()
    density_contours_plot()
    si_plot()
    burning_regions_plot()
    hybrid_burning_regions_plot()
    multipanel()
