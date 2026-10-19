import mkipp2

k = mkipp2.KippenhahnDiagram(logs_dirs=["LOGS-mesa-26.04.1.zip"],
                             identifier="net_nuclear_energy",
                             log10_on_data=False)
k.plot(core_masses=["He", "CO", "ONe", "Fe"],
       save_filename="Kipp2.png")
