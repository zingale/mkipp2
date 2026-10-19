import numpy as np

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

    # only need to read star_age column first
    history = mkipp2.mesa_data.MesaData("LOGS-mesa-26.04.1.zip/history.data",
                                        read_data_cols=["star_age"])
    max_age = max(history.get("star_age"))

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="star_age",
                                 time_units="yr",
                                 identifier="net_nuclear_energy",
                                 function_on_xaxis=lambda x: np.log10(max_age+1.e-5 - x),
                                 log10_on_data=False)

    k.plot(core_masses=["He", "CO", "ONe", "Fe"],
           size=(8, 6),
           save_filename="Kipp-collapse.png")


def density_contours_plot():

    # only need to read star_age column first
    history = mkipp2.mesa_data.MesaData("LOGS-mesa-26.04.1.zip/history.data",
                                        read_data_cols=["star_age"])
    max_age = max(history.get("star_age"))

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="star_age",
                                 time_units="yr",
                                 identifier="net_nuclear_energy",
                                 function_on_xaxis=lambda x: np.log10(max_age+1.e-5 - x),
                                 log10_on_data=False)

    k.plot(core_masses=["He", "CO", "ONe", "Fe"],
           density_contours=[1.e2, 1.e4, 1.e6, 1.e8],
           size=(8, 6),
           save_filename="Kipp-collapse-density.png")


def si_plot():

    # only need to read star_age column first
    history = mkipp2.mesa_data.MesaData("LOGS-mesa-26.04.1.zip/history.data",
                                        read_data_cols=["star_age"])
    max_age = max(history.get("star_age"))

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="star_age",
                                 time_units="yr",
                                 identifier="si28",
                                 function_on_xaxis=lambda x: np.log10(max_age+1.e-5 - x),
                                 log10_on_data=True)

    k.plot(core_masses=["He", "CO", "ONe", "Fe"],
           density_contours=[1.e2, 1.e4, 1.e6, 1.e8],
           size=(8, 6),
           levels=np.arange(-6, 0.5, 0.5),
           save_filename="Kipp-si.png")


def burning_regions_plot():

    # only need to read star_age column first
    history = mkipp2.mesa_data.MesaData("LOGS-mesa-26.04.1.zip/history.data",
                                        read_data_cols=["star_age"])
    max_age = max(history.get("star_age"))

    k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                                 xaxis="star_age",
                                 time_units="yr",
                                 load_profiles=False,
                                 function_on_xaxis=lambda x: np.log10(max_age+1.e-5 - x))

    k.plot_burning_regions(core_masses=["He", "CO", "ONe", "Fe"],
                           size=(8, 6),
                           save_filename="Kipp-burning_regions.png")


if __name__ == "__main__":

    basic_plot()
    age_plot()
    collapse_plot()
    density_contours_plot()
    si_plot()
    burning_regions_plot()
