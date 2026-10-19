import mkipp2

args = mkipp2.KippArgs(
    logs_dirs=["LOGS-mesa-26.04.1.zip"],
    identifier="net_nuclear_energy", log10_on_data=False,
)
mkipp2.kipp_plot(args,
                 core_masses=["He", "CO", "ONe", "Fe"],
                 save_filename="Kipp2.png")
