"""Create Kippenhahn diagrams from MESA history and profile output.

Requirements: history.data and profiles.data containing
              History (star_age,model_number,star_mass,photosphere_r,mixing_regions,mix_relr_regions)
              Profile (mass,radius,eps_nuc)
"""
from collections import namedtuple
import re

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


class KippenhahnDiagram:
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
        by `get_mixing_zones`.

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
                 extractor=None,
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
        if extractor is None:
            self.extractor = self.default_extractor
        else:
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

        self.xyz_data = self.get_xyz_data()
        self.mixing_zones = self.get_mixing_zones()

    def default_extractor(self, identifier, log10_on_data, prof,
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

    def get_mixing_zones(self):
        """Construct mixing-region paths and stellar surface coordinates.

        Returns
        -------
        MixingZones
            Named tuple with ``zones`` (Matplotlib paths), corresponding
            ``mix_types``, transformed ``x_coords``, surface ``y_coords`` (mass,
            radius in solar units), and the loaded ``histories``.
            Surface coordinates and histories retain the full input range even
            when ``xlims`` restricts block construction.

        Raises
        ------
        ValueError
            If history files have different data column names.

        Notes
        -----
        Each layer at sample i becomes a rectangle spanning samples i-1 to i.
        Mass layers use ``mix_type_*`` and ``mix_qtop_*``; radius layers use
        ``mix_relr_type_*`` and ``mix_relr_top_*``. Each layer's
        lower boundary is the preceding layer's top, or zero for the first layer.
        Layers with mixing codes 0 or -1, or below the thickness tolerance, are
        skipped. Adjacent layers with the same mixing code are joined through
        their overlapping vertical intervals into :class:`Zone` boundaries.

        """
        xaxis_divide = self.xaxis_divide
        if self.xaxis == "star_age":
            if self.time_units == "1000 yr":
                xaxis_divide = 1000
            elif self.time_units == "Myr":
                xaxis_divide = 1e6
            elif self.time_units == "Gyr":
                xaxis_divide = 1e9

        if self.yaxis == "radius":
            type_prefix, top_prefix = "mix_relr_type_", "mix_relr_top_"
        else:
            type_prefix, top_prefix = "mix_type_", "mix_qtop_"

        required = {"model_number", "star_age", "star_mass", "log_R", self.xaxis}
        required.update(self.extra_history_cols)

        print("Reading history data")
        mix_data = []
        histories = []
        reference_columns = None
        reference_path = None
        for history_name in self.history_paths:
            history = md.MesaData(history_name, read_data=False)
            history_columns = set(history.columns)
            if reference_columns is None:
                reference_columns = history_columns
                reference_path = history_name
            elif history_columns != reference_columns:
                missing = sorted(reference_columns - history_columns)
                extra = sorted(history_columns - reference_columns)
                raise ValueError(
                    f"History file {str(history_name)!r} has different columns from "
                    f"{str(reference_path)!r}: missing columns {missing}; extra columns {extra}. "
                    "All history files must have the same data column names."
                )
            columns = [
                key for key in history.columns
                if key in required or
                key.endswith("core_mass") or
                re.fullmatch(re.escape(type_prefix) + r"\d+", key) or
                re.fullmatch(re.escape(top_prefix) + r"\d+", key)
            ]
            history.read_data(columns, clean_data=self.clean_data)
            histories.append(history)
        x_coords = []
        for history in histories:
            x_coords.extend(history.get(self.xaxis) / xaxis_divide)
        x_coords = self.function_on_xaxis(np.array(x_coords))
        y_coords = []
        if self.yaxis == "radius":
            for history in histories:
                y_coords.extend(np.power(10, history.get('log_R')))
        else:
            for history in histories:
                y_coords.extend(history.get('star_mass'))

        print("Constructing mixing regions")
        mesa_mix_zones = 0
        while True:
            try:
                mesa_mix_zones = mesa_mix_zones + 1
                mix_type = []
                mix_top = []
                for history in histories:
                    mix_type.extend(history.get(f"{type_prefix}{mesa_mix_zones}"))
                    mix_top.extend(history.get(f"{top_prefix}{mesa_mix_zones}"))
                mix_data.append([mix_type, mix_top])
            except Exception:
                # reached all mix zones included
                mesa_mix_zones = mesa_mix_zones - 1
                print("there are " + str(mesa_mix_zones) + " mixing zones")
                break

        if self.yaxis == "radius":
            tolerance = self.radius_tolerance
        else:
            tolerance = self.mass_tolerance

        zones = []
        mix_types = []
        open_zones = []
        new_zones = []
        for i in range(1, len(x_coords)):
            current_x = x_coords[i]
            # ignore points outside of range, but include one outside each
            # boundary do not assume x_coords is in increasing order
            if self.xlims is not None:
                if not self.xlims[0] <= current_x <= self.xlims[1]:
                    if not ((i+1 < len(x_coords) and
                             self.xlims[0] <= x_coords[i+1] <= self.xlims[1]) or
                            (i-1 > 0 and self.xlims[0] <= x_coords[i-1] <= self.xlims[1])):
                        continue
            previous_x = x_coords[i-1]
            for j in range(0, mesa_mix_zones):
                mix_type = mix_data[j][0][i]
                if mix_type in (0, -1):
                    continue
                max_y_coord = mix_data[j][1][i] * y_coords[i]
                min_y_coord = 0
                if j > 0:
                    min_y_coord = mix_data[j-1][1][i] * y_coords[i]
                # ignore too small regions
                if max_y_coord - min_y_coord < tolerance*y_coords[i]:
                    continue
                zone_block = kipp.ZoneBlock(previous_x, current_x, min_y_coord, max_y_coord)
                exists = False
                zones_to_merge = []
                for z in open_zones:
                    if z.extend(zone_block.vertices[1], zone_block.vertices[1].prev_vertex,
                                mix_type):
                        exists = True
                        z.new_blocks.append(zone_block)
                        zones_to_merge.append(z)
                # merge zones as needed
                for k in range(1, len(zones_to_merge)):
                    zones_to_merge[0].merge_zone(zones_to_merge[k])
                    open_zones.remove(zones_to_merge[k])
                # create zone if it has no predecessor
                if not exists:
                    z = kipp.Zone(zone_block, mix_type)
                    new_zones.append(z)
            open_zones.extend(new_zones)
            new_zones = []
            # separate zones which didn't continue here so we don't need
            # to check them all the time
            for z in open_zones:
                z.switch_new_blocks()
            for z in list(open_zones):
                if len(z.last_blocks) == 0:
                    zones.append(z.get_path())
                    mix_types.append(z.mix_type)
                    open_zones.remove(z)
            # order zones
            temp_open_zones = []
            while len(open_zones) > 0:
                min_y = -1
                j = -1
                for i, zone in enumerate(open_zones):
                    for block in zone.last_blocks:
                        if block.vertices[0].coords[1] < min_y or min_y < 0:
                            min_y = block.vertices[0].coords[1]
                            j = i
                temp_open_zones.append(open_zones.pop(j))
            open_zones = temp_open_zones

        for z in open_zones:
            z.switch_new_blocks()
            zones.append(z.get_path())
            mix_types.append(z.mix_type)

        MixingZones = namedtuple('MixingZones', 'zones mix_types x_coords y_coords histories')
        return MixingZones(zones, mix_types, x_coords, y_coords, histories)

    def get_xyz_data(self):
        """Interpolate a profile quantity onto a common vertical contour grid.

        Returns
        -------
        XYZData
            Named tuple containing ``xlims`` (the minimum and maximum selected
            x coordinates) and ``X``, ``Y``, ``Z`` arrays of shape
            ``(yresolution, number_of_selected_profiles)``. Each column
            represents one profile.

        Raises
        ------
        AssertionError
            If fewer than two profiles remain after header reading and selection.
        RuntimeError
            If a profile header or selected profile body cannot be read.
            The exception identifies the file and preserves the original cause.

        Notes
        -----
        Read headers first to determine the global vertical extent and select
        profiles, then load the required columns. The extent uses all readable
        headers, including those outside ``xlims``. Reverse the profile arrays
        from surface-to-center order before interpolating with ``np.interp``.
        Values outside each profile's coordinate range use endpoint values;
        points above the stellar surface are not masked here.

        """
        xaxis_divide = self.xaxis_divide
        if self.xaxis == "star_age":
            if self.time_units == "1000 yr":
                xaxis_divide = 1000
            elif self.time_units == "Myr":
                xaxis_divide = 1e6
            elif self.time_units == "Gyr":
                xaxis_divide = 1e9

        # Extract data from profiles
        max_x_coord = float('-inf')
        min_x_coord = float('+inf')

        # first read all headers to determine max value on yaxis
        max_y = 0
        print("Reading profile data")

        # read headers to determine required range in y coordinate
        prof_include = [False] * len(self.profile_paths)
        for i, profile_name in enumerate(self.profile_paths):
            try:
                prof = md.MesaData(profile_name, only_read_header=True)
                if self.yaxis == "mass":
                    max_y = max(prof.header['star_mass'], max_y)
                elif self.yaxis == "radius":
                    max_y = max(prof.header['photosphere_r'], max_y)
                # if given xlims, filter out profiles out of range add one
                # profile to each side so data is shown at borders
                if self.xlims is not None:
                    x_coord = self.function_on_xaxis(prof.header[self.xaxis] / xaxis_divide)
                    if self.xlims[0] <= x_coord <= self.xlims[1]:
                        prof_include[i] = True
                        if i > 0:
                            prof_include[i-1] = True
                        if i+1 < len(self.profile_paths):
                            prof_include[i+1] = True
                else:
                    prof_include[i] = True
            except Exception as e:
                raise RuntimeError(f"Couldn't read profile {profile_name}: {e}") from e

        # Filter out profiles outside the selected range and its neighbors.
        profile_paths = [pp for (pp, pi) in zip(self.profile_paths, prof_include) if pi]
        assert len(profile_paths) >= 2, (
            "At least 2 profiles are required after header reading and selection; "
            f"got {len(profile_paths)}."
        )

        # array to interpolate data in the yaxis
        y_interp = np.array([max_y * j / (self.yresolution-1) for j in range(self.yresolution)])

        # now read the data
        # "columns" is a list with the necessary columns that will be
        # parsed from the profile*.data files
        columns = []
        if self.yaxis == "mass":
            columns.append('mass')
        elif self.yaxis == "radius":
            columns.append('radius')

        # call to extractor with return_data_columns = True only gives the
        # required profile columns for the plot

        # actual data is read later with the call to prof.read_data(columns)
        columns.extend(self.extractor(self.identifier,
                                           self.log10_on_data,
                                           None, return_data_columns=True))

        # Initialize interpolation grid
        Z_data_array = np.zeros((self.yresolution, len(profile_paths)))

        # XY coordinates for data
        X_data_array = np.zeros((self.yresolution, len(profile_paths)))
        Y_data_array = np.zeros((self.yresolution, len(profile_paths)))
        for j in range(self.yresolution):
            Y_data_array[j, :] = max_y * j / (self.yresolution-1)
        for i, profile_name in enumerate(profile_paths):
            try:
                prof = md.MesaData(profile_name, read_data=False)
                prof.read_data(columns)
            except Exception as e:
                raise RuntimeError(f"Couldn't read profile {profile_name}: {e}") from e
            x_coord = self.function_on_xaxis(prof.header[self.xaxis] / xaxis_divide)
            if x_coord < max_x_coord:
                print("Profiles are not increasing in X coordinate!!!")
            max_x_coord = max(max_x_coord, x_coord)
            min_x_coord = min(min_x_coord, x_coord)
            if self.yaxis == "mass":
                _ = prof.header['star_mass']
            elif self.yaxis == "radius":
                _ = prof.header['photosphere_r']

            # fill up X positions
            X_data_array[:, i] = x_coord

            # read and interpolate data
            if self.yaxis == "mass":
                y_data = prof.get('mass')
            elif self.yaxis == "radius":
                y_data = prof.get('radius')
            # reverse y_data and z_data for np.interp
            y_data = y_data[::-1]
            z_data = self.extractor(self.identifier, self.log10_on_data, prof)
            z_data = z_data[::-1]
            interp_z_data = np.interp(y_interp, y_data, z_data)

            # set nans outside of range so there is no plotting (did not
            # work, just plot a white region to cover things)

            #for j in range(self.yresolution):
            #    if (Y_data_array[j,i] > prof_y):
            #        Z_data_array[j,i] = np.nan
            #    else:
            Z_data_array[:, i] = interp_z_data

        XYZData = namedtuple('XYZData', 'xlims X Y Z')
        return XYZData((min_x_coord, max_x_coord),
                       X_data_array, Y_data_array, Z_data_array)

    def plot(self, *,
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
        xaxis_divide = self.xaxis_divide
        if self.xaxis == "star_age":
            if self.time_units == "1000 yr":
                xaxis_divide = 1000
            elif self.time_units == "Myr":
                xaxis_divide = 1e6
            elif self.time_units == "Gyr":
                xaxis_divide = 1e9

        if axis is None:
            fig, axis = plt.subplots()
        else:
            fig = axis.figure

        xyz_data = self.xyz_data

        if levels is None:
            levels = []

        if level_style is None:
            level_style = "log"
            if self.identifier == "net_nuclear_energy":
                level_style = "symmetric"

        if core_masses is None:
            core_masses = ["He", "C", "O"]

        if contour_colormap is None:
            if self.identifier == "net_nuclear_energy":
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
                    if self.identifier == "eps_nuc":
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

        mixing_zones = self.mixing_zones
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
        if self.yaxis == "mass":
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
                        self.function_on_xaxis(history.get(self.xaxis) / xaxis_divide),
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
            label = self.identifier
            if self.identifier == "net_nuclear_energy":
                label = (r"$\mathrm{sgn}(\epsilon_{\rm net})\,"
                         r"\log_{10}\max(1, |\epsilon_{\rm net}|)$"
                         "\n" + r"$\epsilon_{\rm net}$ in erg g$^{-1}$ s$^{-1}$")
            elif self.identifier == "eps_nuc":
                if self.log10_on_data:
                    label = r"$\log_{10} |\epsilon_{\rm{nuc}}|$"
                else:
                    label = r"$\epsilon_{\rm{nuc}}$"

            cbar.set_label(label)

            # add axis labels
            if self.xaxis == "star_age":
                axis.set_xlabel(r'$t$ ('+self.time_units+')')
            else:
                axis.set_xlabel(r'$Model Number$')
            if self.yaxis == "radius":
                axis.set_ylabel(r'$r/R_\odot$')
            else:
                axis.set_ylabel(r'$m/M_\odot$')

        if self.xlims is not None:
            axis.set_xlim(self.xlims)

        if show_plot:
            fig.show()
        if save_file:
            fig.tight_layout()
            fig.savefig(save_filename)

        KippPlot = namedtuple('KippPlot', 'contour_plot histories xlims')

        return KippPlot(contour_plot, mixing_zones.histories, xyz_data.xlims)
