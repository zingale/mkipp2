"""Create Kippenhahn diagrams from MESA history and profile output.

Requirements: history.data and profiles.data containing
              History (star_age,model_number,star_mass,photosphere_r,mixing_regions,mix_relr_regions)
              Profile (mass,radius,eps_nuc)
"""
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, PathPatch

import mkipp2.kipp_data as kipp
import mkipp2.mesa_data as md


# Create contour levels array (TBD: Need to be improved)
def get_levels_linear(lmin, lmax, n_levels):
    """Generate evenly spaced contour levels around a data range.

    Parameters
    ----------
    lmin, lmax : float
        Minimum and maximum data values, with ``lmax > lmin``.
    n_levels : int
        Positive number of intervals used to set the spacing.

    Returns
    -------
    numpy.ndarray
        Levels from ``lmin - delta`` up to, but excluding, ``lmax + delta``,
        where ``delta = (lmax - lmin) / n_levels``. Floating-point rounding
        can affect the endpoint and count.
    """
    delta = (lmax - lmin)/n_levels
    levels = np.arange(lmin-delta, lmax+delta, delta)
    return levels


def get_levels_log(lmin, lmax, n_levels):
    """Generate evenly spaced contour levels for logarithmic data.

    Parameters
    ----------
    lmin : float
        Minimum finite logarithmic data value.
    lmax : float
        Maximum finite logarithmic data value, at least ``lmin``.
    n_levels : int
        Positive integer number of intervals.

    Returns
    -------
    numpy.ndarray
        ``n_levels + 1`` levels spanning the bounds rounded outward to
        integers. A constant integer range expands by 0.5 on each side.

    Raises
    ------
    ValueError
        If the interval count is not a positive integer or the bounds
        are not finite and ordered.

    Notes
    -----
    Input values are already logarithms. This function does not transform
    data or generate logarithmically spaced values in linear units.
    """
    if (isinstance(n_levels, (bool, np.bool_)) or
            not isinstance(n_levels, (int, np.integer)) or n_levels < 1):
        raise ValueError("n_levels must be a positive integer")
    if not np.isfinite(lmin) or not np.isfinite(lmax) or lmax < lmin:
        raise ValueError("contour bounds must be finite and ordered")
    lo, hi = np.floor(lmin), np.ceil(lmax)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return np.linspace(lo, hi, n_levels + 1)


def default_extractor(identifier, log10_on_data, prof,
                      return_data_columns=False):
    """Extract a profile column or report the column required to read it.

    Parameters
    ----------
    identifier : str
        Profile column name.
    log10_on_data : bool
        If True, return ``log10(abs(data) + 1e-99)``.
    prof : mesa_data.mesa_data or None
        Loaded profile. May be None when requesting column names.
    return_data_columns : bool, optional
        Return required column names instead of data. Default is False.

    Returns
    -------
    numpy.ndarray or list of str
        Extracted values, optionally transformed, or ``[identifier]`` when
        ``return_data_columns`` is True.
    """
    if return_data_columns:
        return [identifier]
    if log10_on_data:
        return np.log10(abs(prof.get(identifier))+1e-99)
    return prof.get(identifier)


class KippArgs:
    """Configure a Kippenhahn plot and load its profile and mixing data.

    Parameters
    ----------
    logs_dirs : sequence of str, optional
        MESA LOGS directories or ZIP archives used to discover input files.
        Archives may contain the files at their root or in one folder.
        Use ``run.zip/path/to/LOGS`` to select a folder explicitly. Defaults to
        ``['LOGS']``.
    profile_paths : sequence of str, optional
        Ordered profile paths. If None or empty, discover them from
        ``profiles.index`` in each LOGS directory.
    history_paths : sequence of str, optional
        History paths in concatenation order. If None or empty, use
        ``history.data`` in each LOGS directory. All history files must have
        the same data column names; column order may differ.
    clean_data : bool, optional
        Remove history rows superseded by backups or restarts. Default is
        True.
    extra_history_cols : sequence of str, optional
        Additional history columns to load. Defaults to no extra columns.
    identifier : str, optional
        Quantity identifier passed to the extractor and used as the colorbar
        label. The default extractor treats it as a column name. Defaults
        to ``'eps_nuc'``.
    extractor : callable, optional
        Function called as ``extractor(identifier, log10_on_data, prof)``
        to obtain an array of profile values. It must also accept
        ``prof=None, return_data_columns=True`` and return required column
        names. Defaults to `default_extractor`.
    log10_on_data : bool, optional
        Request logarithmic values from the extractor. Default is True.
        The default extractor computes ``log10(abs(data) + 1e-99)``.
        Note: for special fields like "net_nuclear_energy", this is disabled
        since the log is already taken.
    xaxis : {'model_number', 'star_age'}, optional
        Horizontal coordinate. Defaults to ``'model_number'``.
    xaxis_divide : float, optional
        Nonzero divisor applied before the x-axis transform. Defaults to
        1.0. Certain ``time_units`` override it when plotting ``star_age``.
    time_units : {'yr', '1000 yr', 'Myr', 'Gyr'}, optional
        Time-axis label, defaulting to ``'Myr'``. For ``star_age``, the last
        three choices override the divisor with 1000, 1e6, and 1e9,
        respectively. ``'yr'`` retains ``xaxis_divide``.
    function_on_xaxis : callable, optional
        Transform applied after division. Must accept scalar profile ages
        or model numbers and NumPy arrays from histories. Defaults to the
        identity function.
    yaxis : {'mass', 'radius'}, optional
        Vertical coordinate, in solar masses or solar radii. Defaults to
        ``'mass'``.
    yresolution : int, optional
        Number of vertical interpolation points; must be at least two.
        Defaults to 1000.
    mass_tolerance, radius_tolerance : float, optional
        Minimum mixing-layer thickness as a fraction of the surface
        coordinate, using the tolerance for the selected y axis. Each
        defaults to 1e-7.
    xlims : tuple of float, optional
        Inclusive limits in transformed x coordinates, used for data
        selection and displayed axis limits. Neighboring samples are kept
        for boundary coverage. Defaults to the full range.

    Attributes
    ----------
    xyz_data : namedtuple
        Interpolated profile data with fields ``xlims``, ``X``, ``Y``, and
        ``Z``, returned by `kipp_data.get_xyz_data`.
    mixing_zones : namedtuple
        Mixing-region paths and history data with fields ``zones``,
        ``mix_types``, ``x_coords``, ``y_coords``, and ``histories``, returned
        by `kipp_data.get_mixing_zones`.

    Notes
    -----
    All parameters are keyword-only and are stored as instance attributes,
    with defaults resolved as described above. Construction immediately
    reads the input files and computes both data products; changing input
    settings afterward does not recompute them.

    """

    def __init__(self, *,
                 logs_dirs=None,
                 profile_paths=None,
                 history_paths=None,
                 clean_data=True,
                 extra_history_cols=None,
                 identifier="eps_nuc",
                 extractor=default_extractor,
                 log10_on_data=True,
                 xaxis="model_number",
                 xaxis_divide=1.0,
                 time_units="Myr",
                 function_on_xaxis=lambda x: x,
                 yaxis="mass",
                 yresolution=1000,
                 mass_tolerance=0.0000001,
                 radius_tolerance=0.0000001,
                 xlims=None):
        """Store plot settings and prepare profile and mixing-region data."""

        if logs_dirs is None:
            logs_dirs = ['LOGS']

        self.logs_dirs = [md.resolve_logs_dir(log_dir) for log_dir in logs_dirs]
        self.clean_data = clean_data

        if extra_history_cols is None:
            extra_history_cols = []
        self.extra_history_cols = extra_history_cols

        self.identifier = identifier
        self.extractor = extractor
        self.log10_on_data = log10_on_data

        # override for fields where the log is done by MESA
        if self.identifier == "net_nuclear_energy":
            self.log10_on_data = False

        self.xaxis = xaxis
        self.xaxis_divide = xaxis_divide
        self.time_units = time_units
        self.function_on_xaxis = function_on_xaxis
        self.yaxis = yaxis

        self.yresolution = yresolution
        self.mass_tolerance = mass_tolerance
        self.radius_tolerance = radius_tolerance
        self.xlims = xlims

        # Fill profile and history names if unspecified
        if profile_paths is None:
            profile_paths = []
        self.profile_paths = profile_paths
        if len(self.profile_paths) == 0:
            self.profile_paths = md.get_profile_paths(logs_dirs=self.logs_dirs)

        if history_paths is None:
            history_paths = []
        self.history_paths = history_paths
        if len(self.history_paths) == 0:
            self.history_paths = []
            for log_dir in self.logs_dirs:
                self.history_paths.append(log_dir+"/history.data")

        self.xyz_data = kipp.get_xyz_data(self.profile_paths, self, xlims=xlims)
        self.mixing_zones = kipp.get_mixing_zones(self.history_paths,
                                                  self, xlims=xlims)


def kipp_plot(kipp_args, *,
              axis=None,
              contour_colormap=None,
              core_masses=None,
              decorate_plot=True,
              levels=None,
              level_style=None,
              num_levels=8,
              show_mixing_legend=True,
              show_core_mass_legend=True,
              show_plot=False,
              show_conv=True,
              show_therm=True,
              show_semi=True,
              show_over=True,
              show_rot=False,
              save_file=True,
              save_filename="Kippenhahn.png"):
    """Draw profile contours, mixing boundaries, and stellar structure.

    Parameters
    ----------
    kipp_args : KippArgs
        Prepared data and plot settings. Controls contour levels, visible
        mixing types, core-mass overlays, labels, display, and file output.
    axis : matplotlib.axes.Axes, optional
        Axes on which to draw. If None, create a figure and axes.
    contour_colormap : matplotlib.colors.Colormap, optional
        Colormap for the filled contours. Defaults to ``'Blues'``
        for most data.  For diverging fields, like ``'net_nuclear_energy'``,
        the default is ``RdBu_r``.
    core_masses : sequence of {'He', 'C', 'O', 'CO', 'ONe', 'Fe'}, optional
        Core masses to overlay when ``yaxis='mass'``. Defaults to He, C, O;
        an empty sequence suppresses them.
    decorate_plot : bool, optional
        Add axis labels and a colorbar.  Default is True.
    levels : array_like, optional
        Explicit, increasing contour levels. If None or empty, generate
        levels from the extracted data.  For some fields, like "eps_nuc",
        we clip the data to show only values > 1.
    level_style : str, optional
        If ``levels`` is not provided, then we can compute the levels
        as "log" (evenly spaced levels in already-logarithmic data), "linear", or
        "symmetric" (centered on zero, and adapting to data range).
        Default is "log" for most data, but "symmetric" for data that
        is diverging (like "net_nuclear_energy").
    num_levels : int, optional
        Positive number of intervals used by the linear and logarithmic
        level generators. Defaults to 8; ignored for symmetric levels.
    show_conv, show_therm, show_semi, show_over : bool, optional
        Display convection, thermohaline mixing, semiconvection, and
        overshooting boundaries, respectively. Each defaults to True.
    show_mixing_legend : bool, optional
        Include mixing types in the first legend column. Default is True.
    show_core_mass_legend : bool, optional
        Include core-mass lines in the second legend column. Default is True.
    show_plot : bool, optional
        Call ``pyplot.show()`` after plotting. Default is False.
    show_rot : bool, optional
        Display rotational mixing boundaries. Default is False.
    save_file : bool, optional
        Save the current pyplot figure after the optional show call.
        Default is True.
    save_filename : str, optional
        Output filename; its extension determines the format. Defaults to
        ``'Kippenhahn.png'``.

    Returns
    -------
    KippPlot
        Named tuple containing ``contour_plot`` (the contour set returned
        by ``Axes.contourf``), ``histories`` (loaded ``mesa_data`` objects),
        and ``xlims`` (the extent of the selected profile coordinates,
        which can extend beyond the requested plotting limits).

    Notes
    -----
    Mixing regions are drawn as unfilled, hatched patches over the profile
    contours. A white fill covers contours above the stellar surface, and
    selected core masses are overlaid for a mass axis. Axis labels and a
    colorbar are added only when ``decorate_plot`` is True. Mixing and
    core-mass legend entries are controlled independently by their respective
    ``show_*_legend`` options. Each displayed type appears once in a shared
    two-column legend, with mixing types first and core masses second.
    When only one group is shown, the legend uses a single column.

    Display and file output use pyplot. In particular,
    ``save_file=True`` saves the current pyplot figure, which need not
    be the figure owning a supplied ``axis``. File output occurs after
    the optional show call.

    """
    xaxis_divide = kipp_args.xaxis_divide
    if kipp_args.xaxis == "star_age":
        if kipp_args.time_units == "1000 yr":
            xaxis_divide = 1000
        elif kipp_args.time_units == "Myr":
            xaxis_divide = 1e6
        elif kipp_args.time_units == "Gyr":
            xaxis_divide = 1e9

    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.figure

    xyz_data = kipp_args.xyz_data

    if levels is None:
        levels = []

    if level_style is None:
        level_style = "log"
        if kipp_args.identifier == "net_nuclear_energy":
            level_style = "symmetric"

    if core_masses is None:
        core_masses = ["He", "C", "O"]

    if contour_colormap is None:
        if kipp_args.identifier == "net_nuclear_energy":
            contour_colormap = plt.get_cmap("RdBu_r")
        else:
            contour_colormap = plt.get_cmap("Blues")

    # only plot if there is data
    if xyz_data.Z.size > 0:
        # Get levels if undefined
        if len(levels) == 0:
            if level_style == "log":
                finite = xyz_data.Z[np.isfinite(xyz_data.Z)]
                if finite.size == 0:
                    raise ValueError("no finite profile values available for contour levels")
                lmin = finite.min()
                # overrides for some fields here
                if kipp_args.identifier == "eps_nuc":
                    lmin = 0.0
                levels = get_levels_log(lmin, finite.max(), num_levels)
            elif level_style == "linear":
                levels = get_levels_linear(np.nanmin(xyz_data.Z[:, :]),
                                           np.nanmax(xyz_data.Z[:, :]),
                                           num_levels)
            elif level_style == "symmetric":
                data_limit = max(1, int(np.ceil(np.nanmax(np.abs(xyz_data.Z)))))
                levels = np.arange(-data_limit, data_limit + 1)
            else:
                raise ValueError("invalid level_style")

        # make plot
        contour_plot = axis.contourf(xyz_data.X, xyz_data.Y, xyz_data.Z[:, :],
                                     cmap=contour_colormap,
                                     levels=levels, antialiased=False)
    else:
        print("No profile data to plot!")
        contour_plot = axis.contourf([[None, None]], [[None, None]], [[None, None]],
                                     cmap=contour_colormap,
                                     antialiased=False)

    mixing_zones = kipp_args.mixing_zones
    mixing_handles = {}

    for i, zone in enumerate(mixing_zones.zones):
        color = ""
        # Convective mixing
        if mixing_zones.mix_types[i] == 1 and show_conv:
            label = "Convection"
            color = "Chartreuse"
            hatch = "//"
            line = 1
        # Overshooting
        elif mixing_zones.mix_types[i] == 3 and show_over:
            label = "Overshooting"
            color = "purple"
            hatch = "x"
            line = 1
        # Semiconvective mixing
        elif mixing_zones.mix_types[i] == 4 and show_semi:
            label = "Semiconvection"
            color = "red"
            hatch = "\\\\"
            line = 1
        # Thermohaline mixing
        elif mixing_zones.mix_types[i] == 5 and show_therm:
            label = "Thermohaline"
            color = "Gold"  # Salmon
            hatch = "||"
            line = 1
        # Rotational mixing
        elif mixing_zones.mix_types[i] == 6 and show_rot:
            label = "Rotational"
            color = "brown"
            hatch = "*"
            line = 1
        # Anonymous mixing
        else:
            label = "_nolegend_"
            color = "white"
            hatch = " "
            line = 0
        patch = PathPatch(zone, fill=False, hatch=hatch,
                          edgecolor=color, linewidth=line, label=label)
        axis.add_patch(patch)
        if line:
            mixing_handles.setdefault(mixing_zones.mix_types[i], patch)

    # limit x_coords to data of contours and add line at stellar surface
    for i, x_coord in enumerate(mixing_zones.x_coords):
        if x_coord > xyz_data.xlims[1]:
            break

    # Also fill with white above the plot to cover any remainder of the
    # contour plot
    axis.fill_between(mixing_zones.x_coords, max(mixing_zones.y_coords),
                      mixing_zones.y_coords, color="w")
    axis.plot(mixing_zones.x_coords[:i], mixing_zones.y_coords[:i], "k-")

    # add core masses
    core_mass_handles = {}
    if kipp_args.yaxis == "mass":
        field_name = ""
        for core_mass in core_masses:
            if core_mass == "He":
                field_name = "he_core_mass"
                color = "C0:"
            elif core_mass == "C":
                field_name = "c_core_mass"
                color = "C1:"
            elif core_mass == "O":
                field_name = "o_core_mass"
                color = "C2:"
            elif core_mass == "CO":
                field_name = "co_core_mass"
                color = "C1:"
            elif core_mass == "ONe":
                field_name = "one_core_mass"
                color = "C2:"
            elif core_mass == "Fe":
                field_name = "fe_core_mass"
                color = "C3:"
            for history in mixing_zones.histories:
                core_line, = axis.plot(
                    kipp_args.function_on_xaxis(history.get(kipp_args.xaxis) / xaxis_divide),
                    history.get(field_name), color, lw=2, label=f"{core_mass} core")
                core_mass_handles.setdefault(core_mass, core_line)

    mixing_entries = ([mixing_handles[key] for key in sorted(mixing_handles)]
                      if show_mixing_legend else [])
    core_entries = list(core_mass_handles.values()) if show_core_mass_legend else []
    rows = max(len(mixing_entries), len(core_entries))
    if mixing_entries and core_entries:
        # Matplotlib fills columns in order; pad to keep each group in its column.
        handles = []
        labels = []
        for entries in (mixing_entries, core_entries):
            handles.extend(entries)
            labels.extend(entry.get_label() for entry in entries)
            handles.extend(Patch(visible=False) for _ in range(rows - len(entries)))
            labels.extend([""] * (rows - len(entries)))
        axis.legend(handles=handles, labels=labels, ncol=2,
                    loc="best", fontsize="small")
    elif rows:
        axis.legend(handles=mixing_entries or core_entries,
                    loc="best", fontsize="small")

    if decorate_plot:

        # add colorbar
        cbar = plt.colorbar(contour_plot, pad=0.05)

        # use pretty labels if we know the field
        label = kipp_args.identifier
        if kipp_args.identifier == "net_nuclear_energy":
            label = (r"$\mathrm{sgn}(\epsilon_{\rm net})\,"
                     r"\log_{10}\max(1, |\epsilon_{\rm net}|)$"
                     "\n" + r"$\epsilon_{\rm net}$ in erg g$^{-1}$ s$^{-1}$")
        elif kipp_args.identifier == "eps_nuc":
            if kipp_args.log10_on_data:
                label = r"$\log_{10} |\epsilon_{\rm{nuc}}|$"
            else:
                label = r"$\epsilon_{\rm{nuc}}$"

        cbar.set_label(label)

        # add axis labels
        if kipp_args.xaxis == "star_age":
            axis.set_xlabel(r'$t$ ('+kipp_args.time_units+')')
        else:
            axis.set_xlabel(r'$Model Number$')
        if kipp_args.yaxis == "radius":
            axis.set_ylabel(r'$r/R_\odot$')
        else:
            axis.set_ylabel(r'$m/M_\odot$')

    if kipp_args.xlims is not None:
        axis.set_xlim(kipp_args.xlims)

    if show_plot:
        fig.show()
    if save_file:
        fig.tight_layout()
        fig.savefig(save_filename)

    KippPlot = namedtuple('KippPlot', 'contour_plot histories xlims')

    return KippPlot(contour_plot, mixing_zones.histories, xyz_data.xlims)
