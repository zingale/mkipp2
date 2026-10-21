"""Create Kippenhahn diagrams from MESA history and profile output.

``KippenhahnDiagram`` reads history and profile files at construction and
reuses the prepared data for subsequent calls to ``plot()``. By default,
input paths are discovered from ``history.data`` and ``profiles.index`` in
the LOGS directory. Profile columns depend on the chosen y axis and extractor;
history columns include stellar coordinates, mixing regions, and core masses.
"""
from collections import namedtuple
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch, PathPatch
from matplotlib.ticker import MaxNLocator

import mkipp2.kipp_data as kipp
import mkipp2.mesa_data as md


def format_scientific(value):
    """Format a positive finite value as scientific notation for MathText.

    Parameters
    ----------
    value : float
        Positive finite value to format with six significant digits.

    Returns
    -------
    str
        MathText expression without dollar delimiters or units. A coefficient
        of one is omitted, leaving just the power of ten.
    """
    coefficient, exponent = f"{value:.5e}".split("e")
    coefficient = coefficient.rstrip("0").rstrip(".")
    power = f"10^{{{int(exponent)}}}"
    return power if coefficient == "1" else rf"{coefficient}\times {power}"


def _visible_contour_midpoint(vertices, axis):
    """Return the visible arc midpoint in data coordinates and length in pixels."""
    points = axis.transData.transform(vertices)
    starts = points[:-1]
    steps = np.diff(points, axis=0)
    lower = np.zeros(len(steps))
    upper = np.ones(len(steps))
    valid = np.all(np.isfinite(starts) & np.isfinite(steps), axis=1)
    # Clip each line segment to the axes rectangle, including segments whose
    # endpoints both lie outside it.
    for dim, bounds in enumerate(((axis.bbox.xmin, axis.bbox.xmax),
                                  (axis.bbox.ymin, axis.bbox.ymax))):
        moving = steps[:, dim] != 0
        valid &= moving | ((starts[:, dim] >= bounds[0]) & (starts[:, dim] <= bounds[1]))
        entry = np.zeros(len(steps))
        leave = np.ones(len(steps))
        entry[moving] = (bounds[0] - starts[moving, dim]) / steps[moving, dim]
        leave[moving] = (bounds[1] - starts[moving, dim]) / steps[moving, dim]
        lower = np.maximum(lower, np.minimum(entry, leave))
        upper = np.minimum(upper, np.maximum(entry, leave))
    valid &= upper > lower
    starts = starts[valid] + lower[valid, None] * steps[valid]
    steps = (upper - lower)[valid, None] * steps[valid]
    lengths = np.linalg.norm(steps, axis=1)
    total = lengths.sum()
    if total == 0:
        return None, 0
    cumulative = np.cumsum(lengths)
    index = np.searchsorted(cumulative, total / 2)
    fraction = (total / 2 - (cumulative[index] - lengths[index])) / lengths[index]
    midpoint = starts[index] + fraction * steps[index]
    return axis.transData.inverted().transform(midpoint), total


def default_extractor(identifier, log10_on_data, prof,
                      return_data_columns=False):
    """Extract a profile column or report the column required to read it.

    Parameters
    ----------
    identifier : str
        Profile column name.
    log10_on_data : bool
        If True, return ``log10(abs(data) + 1e-99)``.
    prof : mkipp2.mesa_data.MesaData or None
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
    logs_dirs : sequence of str or path-like, optional
        MESA LOGS directories or ZIP archives used to discover input files.
        Archives may contain the files at their root or in one folder.
        Use ``run.zip/path/to/LOGS`` to select a folder explicitly. Defaults to
        ``['LOGS']``.
    profile_paths : sequence of str or path-like, optional
        Ordered profile paths. If None or empty, discover them from
        ``profiles.index`` in each LOGS directory.
    history_paths : sequence of str or path-like, optional
        History paths in concatenation order. If None or empty, use
        ``history.data`` in each LOGS directory. All history files must have
        the same data column names; column order may differ.
    load_profiles : bool, optional
        Load profile data during construction. Default is True. Set False
        for history-only ``plot_burning_regions()``; ``xyz_data`` is then None
        and neither profile files nor ``profiles.index`` are read.
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
        Forced to False for ``'net_nuclear_energy'``, whose input values are
        treated as already transformed, including when using a custom extractor.
    xaxis : {'model_number', 'star_age', 'collapse'}, optional
        Horizontal coordinate. Defaults to ``'model_number'``.
        For ``'collapse'``, use ``log10((max_age - age) / divisor +
        time_offset)`` and display decreasing values from left to right.
        The final age is the maximum across the input histories. Cannot be
        combined with ``function_on_xaxis``.
    time_offset : float, optional
        Finite, positive offset in the selected time units for the collapse
        transform, preventing log(0). Defaults to 1e-5.
    xaxis_divide : float, optional
        Nonzero divisor applied before the x-axis transform. Defaults to
        1.0. Certain ``time_units`` override it for ``star_age`` and ``collapse``.
        Must be finite and positive for ``collapse``.
    time_units : {'yr', '1000 yr', 'Myr', 'Gyr'}, optional
        Time-axis label, defaulting to ``'Myr'``. For ``star_age`` and ``collapse``, the last
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
    xlims : sequence of two floats, optional
        Inclusive limits in transformed x coordinates, used for data
        selection and displayed axis limits, supplied in increasing order.
        Collapse mode reverses these limits for display. Neighboring samples
        are kept for boundary coverage. Defaults to the full range.

    Attributes
    ----------
    min_age, max_age : float or None
        Stellar-age bounds in years across the input histories, respecting
        ``clean_data``. Computed for ``xaxis='collapse'``; otherwise None.
    xyz_data : namedtuple
        Interpolated profile data with fields ``xlims``, ``X``, ``Y``, and
        ``Z``, prepared during construction. Arrays have shape
        ``(yresolution, number_of_selected_profiles)``; ``xlims`` gives the
        extent of selected profile coordinates, including boundary neighbors.
    mixing_zones : namedtuple
        Mixing-region paths and history data with fields ``zones``,
        ``mix_types``, ``x_coords``, ``y_coords``, and ``histories``, prepared
        during construction.

    Notes
    -----
    All parameters are keyword-only and exposed as read-only properties,
    with defaults resolved as described above. Sequence settings are copied
    into tuples. Construction immediately reads the input files and computes
    both data products, except profiles when ``load_profiles=False``.
    Create a new diagram to change input settings;
    rendering options can be changed on each call to ``plot()``.
    The prepared arrays and history objects are not made immutable. Custom
    callables are retained by reference and should keep the same behavior
    throughout the diagram's lifetime.

    Examples
    --------
    Load data once and draw a diagram without saving a file:

    .. code-block:: python

        diagram = KippenhahnDiagram(logs_dirs=["LOGS"], xaxis="star_age")
        result = diagram.plot(save_file=False)

    Access the prepared data through ``diagram.xyz_data`` and
    ``diagram.mixing_zones``. Call ``diagram.plot(axis=another_axis)`` to
    draw the same data on another set of axes.

    """

    def __init__(self, *,
                 logs_dirs=None,
                 profile_paths=None,
                 history_paths=None,
                 clean_data=True,
                 load_profiles=True,
                 extra_history_cols=None,
                 identifier="eps_nuc",
                 extractor=None,
                 log10_on_data=True,
                 xaxis="model_number",
                 time_offset=1.e-5,
                 xaxis_divide=1.0,
                 time_units="Myr",
                 function_on_xaxis=None,
                 yaxis="mass",
                 yresolution=1000,
                 mass_tolerance=0.0000001,
                 radius_tolerance=0.0000001,
                 xlims=None):
        """Store plot settings and prepare profile and mixing-region data."""

        if logs_dirs is None:
            logs_dirs = ['LOGS']

        self._logs_dirs = tuple(md.resolve_logs_dir(log_dir) for log_dir in logs_dirs)
        self._clean_data = clean_data
        self._load_profiles = load_profiles

        if extra_history_cols is None:
            extra_history_cols = []
        self._extra_history_cols = tuple(extra_history_cols)

        self._identifier = identifier
        if extractor is None:
            self._extractor = default_extractor
        else:
            self._extractor = extractor
        self._log10_on_data = log10_on_data

        # override for fields where the log is done by MESA
        if self._identifier == "net_nuclear_energy":
            self._log10_on_data = False

        self._xaxis = xaxis
        self._time_offset = time_offset
        if xaxis == "collapse":
            if function_on_xaxis is not None:
                raise ValueError("Error, function_on_xaxis cannot be specified with xaxis='collapse'")
            if not np.isfinite(time_offset) or time_offset <= 0:
                raise ValueError("time_offset must be finite and positive for xaxis='collapse'")

        self._xaxis_divide = xaxis_divide
        self._time_units = time_units
        if xaxis == "collapse" and (not np.isfinite(self._xaxis_divisor) or
                                    self._xaxis_divisor <= 0):
            raise ValueError("The time divisor must be finite and positive for xaxis='collapse'")

        if function_on_xaxis is None:
            self._function_on_xaxis = lambda x: x
        else:
            self._function_on_xaxis = function_on_xaxis

        self._yaxis = yaxis

        self._yresolution = yresolution
        self._mass_tolerance = mass_tolerance
        self._radius_tolerance = radius_tolerance
        self._xlims = tuple(xlims) if xlims is not None else None

        # Discover unspecified paths and store immutable copies of sequences.
        self._profile_paths = tuple(profile_paths) if profile_paths is not None else ()
        if self.load_profiles and not self.profile_paths:
            self._profile_paths = tuple(md.get_profile_paths(logs_dirs=self.logs_dirs))

        self._history_paths = tuple(history_paths) if history_paths is not None else ()
        if not self.history_paths:
            self._history_paths = tuple(log_dir + "/history.data" for log_dir in self.logs_dirs)

        self._min_age = None
        self._max_age = None
        if xaxis == "collapse":
            histories = [md.MesaData(history_path, read_data_cols=["star_age"],
                                     clean_data=self.clean_data)
                         for history_path in self.history_paths]
            self._max_age = max(np.max(history.get("star_age")) for history in histories)
            self._min_age = min(np.min(history.get("star_age")) for history in histories)

            if not np.isfinite(self.min_age) or not np.isfinite(self.max_age):
                raise ValueError("Collapse mode requires finite history ages")
            self._function_on_xaxis = self._collapse_transform

        self.xyz_data = self._build_xyz_data() if self.load_profiles else None
        self.mixing_zones = self._build_mixing_zones()

    @property
    def min_age(self):
        """Return the minimum history age in years for collapse mode."""
        return self._min_age

    @property
    def max_age(self):
        """Return the maximum history age in years for collapse mode."""
        return self._max_age

    @property
    def time_offset(self):
        """Return the fixed collapse offset in the selected time units."""
        return self._time_offset

    def _collapse_transform(self, age):
        """Transform ages already divided into the selected time units."""
        remaining = self.max_age / self._xaxis_divisor - np.asarray(age) + self.time_offset
        if np.any(~np.isfinite(remaining)) or np.any(remaining <= 0):
            raise ValueError("Collapse coordinates require finite, positive time remaining plus time_offset")
        return np.log10(remaining)

    @property
    def _xaxis_column(self):
        """Return the MESA column underlying the selected horizontal axis."""
        return "star_age" if self.xaxis == "collapse" else self.xaxis

    @property
    def load_profiles(self):
        """Return whether profile data are loaded during construction."""
        return self._load_profiles

    @property
    def logs_dirs(self):
        """Return the fixed ``logs_dirs`` setting."""
        return self._logs_dirs

    @property
    def profile_paths(self):
        """Return the fixed ``profile_paths`` setting."""
        return self._profile_paths

    @property
    def history_paths(self):
        """Return the fixed ``history_paths`` setting."""
        return self._history_paths

    @property
    def clean_data(self):
        """Return the fixed ``clean_data`` setting."""
        return self._clean_data

    @property
    def extra_history_cols(self):
        """Return the fixed ``extra_history_cols`` setting."""
        return self._extra_history_cols

    @property
    def identifier(self):
        """Return the fixed ``identifier`` setting."""
        return self._identifier

    @property
    def extractor(self):
        """Return the fixed ``extractor`` setting."""
        return self._extractor

    @property
    def log10_on_data(self):
        """Return the fixed ``log10_on_data`` setting."""
        return self._log10_on_data

    @property
    def xaxis(self):
        """Return the fixed ``xaxis`` setting."""
        return self._xaxis

    @property
    def xaxis_divide(self):
        """Return the fixed ``xaxis_divide`` setting."""
        return self._xaxis_divide

    @property
    def time_units(self):
        """Return the fixed ``time_units`` setting."""
        return self._time_units

    @property
    def function_on_xaxis(self):
        """Return the fixed ``function_on_xaxis`` setting."""
        return self._function_on_xaxis

    @property
    def yaxis(self):
        """Return the fixed ``yaxis`` setting."""
        return self._yaxis

    @property
    def yresolution(self):
        """Return the fixed ``yresolution`` setting."""
        return self._yresolution

    @property
    def mass_tolerance(self):
        """Return the fixed ``mass_tolerance`` setting."""
        return self._mass_tolerance

    @property
    def radius_tolerance(self):
        """Return the fixed ``radius_tolerance`` setting."""
        return self._radius_tolerance

    @property
    def xlims(self):
        """Return the fixed ``xlims`` setting."""
        return self._xlims

    @property
    def _xaxis_divisor(self):
        """Return the divisor shared by profile, mixing, and core coordinates.

        For stellar age, ``1000 yr``, ``Myr``, and ``Gyr`` override the
        configured divisor. Otherwise, retain ``xaxis_divide``.
        """
        if self.xaxis in ("star_age", "collapse"):
            return {"1000 yr": 1000, "Myr": 1e6, "Gyr": 1e9}.get(
                self.time_units, self.xaxis_divide)
        return self.xaxis_divide

    def _build_mixing_zones(self):
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
        Read histories using the fixed instance settings and return a new
        data product. The constructor stores it as ``mixing_zones``.

        Each layer at sample i becomes a rectangle spanning samples i-1 to i.
        Mass layers use ``mix_type_*`` and ``mix_qtop_*``; radius layers use
        ``mix_relr_type_*`` and ``mix_relr_top_*``. Each layer's
        lower boundary is the preceding layer's top, or zero for the first layer.
        Layers with mixing codes 0 or -1, or below the thickness tolerance, are
        skipped. Adjacent layers with the same mixing code are joined through
        their overlapping vertical intervals into :class:`mkipp2.kipp_data.Zone`
        boundaries.

        """
        xaxis_divide = self._xaxis_divisor

        if self.yaxis == "radius":
            type_prefix, top_prefix = "mix_relr_type_", "mix_relr_top_"
        else:
            type_prefix, top_prefix = "mix_type_", "mix_qtop_"

        required = {"model_number", "star_age", "star_mass", "log_R", self._xaxis_column}
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
                re.fullmatch(r"burn_(?:type_|qtop_|relr_type_|relr_top_)\d+", key) or
                re.fullmatch(re.escape(type_prefix) + r"\d+", key) or
                re.fullmatch(re.escape(top_prefix) + r"\d+", key)
            ]
            history.read_data(columns, clean_data=self.clean_data)
            histories.append(history)
        x_coords = []
        for history in histories:
            x_coords.extend(history.get(self._xaxis_column) / xaxis_divide)
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

        zones, mix_types = self._build_region_paths(
            mix_data, x_coords, y_coords, skip_types=(0, -1), tolerance=tolerance)

        MixingZones = namedtuple('MixingZones', 'zones mix_types x_coords y_coords histories')
        return MixingZones(zones, mix_types, x_coords, y_coords, histories)

    def _build_buring_zones(self):
        """Construct burning-region paths from the loaded history data.

        Returns
        -------
        BurningZones
            Named tuple with ``zones`` (stitched Matplotlib paths),
            ``burn_types`` (signed logarithmic heating/cooling levels),
            ``x_coords``, ``y_coords`` (stellar surface), and ``histories``.

        Raises
        ------
        ValueError
            If histories are absent or burning-region columns are missing,
            unpaired, or not consecutively numbered from one.

        Notes
        -----
        Reuse the histories and coordinates prepared by ``_build_mixing_zones``.
        Build a new data product without rereading files or creating artists.
        Zero and unused (-9999) values are skipped; negative values remain as
        cooling regions. Adjacent equal-valued blocks use the shared stitching
        algorithm, without a minimum layer thickness.
        """
        prefix = "burn_relr_" if self.yaxis == "radius" else "burn_"
        top_prefix = prefix + ("top_" if self.yaxis == "radius" else "qtop_")
        type_prefix = prefix + "type_"
        histories = self.mixing_zones.histories
        if not histories:
            raise ValueError("No histories available for burning regions")
        indices = sorted(int(key[len(type_prefix):]) for key in histories[0].columns
                         if re.fullmatch(re.escape(type_prefix) + r"\d+", key))
        top_indices = sorted(int(key[len(top_prefix):]) for key in histories[0].columns
                             if re.fullmatch(re.escape(top_prefix) + r"\d+", key))
        if not indices or indices != top_indices or indices != list(range(1, max(indices) + 1)):
            raise ValueError(f"Burning regions require consecutive paired {type_prefix}N and {top_prefix}N "
                             f"columns in {histories[0].filename}")
        x = np.asarray(self.mixing_zones.x_coords)
        surface = np.asarray(self.mixing_zones.y_coords)
        region_data = []
        for region in indices:
            types = np.concatenate([history.get(f"{type_prefix}{region}") for history in histories])
            tops = np.concatenate([history.get(f"{top_prefix}{region}") for history in histories])
            region_data.append((types, tops))
        paths, values = self._build_region_paths(
            region_data, x, surface, skip_types=(0, -9999), tolerance=0)
        BurningZones = namedtuple('BurningZones', 'zones burn_types x_coords y_coords histories')
        return BurningZones(paths, values, x, surface, histories)

    def _build_region_paths(self, region_data, x_coords, y_coords, *, skip_types, tolerance):
        """Stitch adjacent equal-valued history layers into connected paths.

        Each entry in region_data contains a type/value array and a fractional
        upper-bound array. Zone tracks splits and merges, preserving holes in
        the resulting compound paths. Skipped values still define the lower
        boundary of the next layer. Intervals outside xlims close open zones.
        """
        zones = []
        region_types = []
        open_zones = []
        new_zones = []
        for i in range(1, len(x_coords)):
            current_x = x_coords[i]
            previous_x = x_coords[i-1]
            include = (self.xlims is None or
                       (max(previous_x, current_x) >= self.xlims[0] and
                        min(previous_x, current_x) <= self.xlims[1]))
            # Empty intervals still close open zones, preventing joins across gaps.
            for j in (range(len(region_data)) if include else ()):
                region_type = region_data[j][0][i]
                if region_type in skip_types:
                    continue
                max_y_coord = region_data[j][1][i] * y_coords[i]
                min_y_coord = 0
                if j > 0:
                    min_y_coord = region_data[j-1][1][i] * y_coords[i]
                max_y_coord = min(max_y_coord, y_coords[i])
                if (not np.all(np.isfinite([previous_x, current_x, min_y_coord, max_y_coord, region_type])) or
                        min_y_coord < 0 or max_y_coord <= min_y_coord):
                    continue
                # ignore too small regions
                if max_y_coord - min_y_coord < tolerance*y_coords[i]:
                    continue
                zone_block = kipp.ZoneBlock(previous_x, current_x, min_y_coord, max_y_coord)
                exists = False
                zones_to_merge = []
                for z in open_zones:
                    if z.extend(zone_block.vertices[1], zone_block.vertices[1].prev_vertex,
                                region_type):
                        exists = True
                        z.new_blocks.append(zone_block)
                        zones_to_merge.append(z)
                # merge zones as needed
                for k in range(1, len(zones_to_merge)):
                    zones_to_merge[0].merge_zone(zones_to_merge[k])
                    open_zones.remove(zones_to_merge[k])
                # create zone if it has no predecessor
                if not exists:
                    z = kipp.Zone(zone_block, region_type)
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
                    region_types.append(z.mix_type)
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
            region_types.append(z.mix_type)

        return zones, region_types

    def _build_xyz_data(self):
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
        Read profiles using the fixed instance settings and return a new
        data product. The constructor stores it as ``xyz_data``.

        Read headers first to determine the global vertical extent and select
        profiles, then load the required columns. The extent uses all readable
        headers, including those outside ``xlims``. Reverse the profile arrays
        from surface-to-center order before interpolating with ``np.interp``.
        Values outside each profile's coordinate range use endpoint values;
        points above the stellar surface are not masked here.

        """
        xaxis_divide = self._xaxis_divisor

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
                    x_coord = self.function_on_xaxis(prof.header[self._xaxis_column] / xaxis_divide)
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
        self._log_density = np.full_like(Z_data_array, np.nan)
        self._missing_density_profiles = []

        # XY coordinates for data
        X_data_array = np.zeros((self.yresolution, len(profile_paths)))
        Y_data_array = np.zeros((self.yresolution, len(profile_paths)))
        for j in range(self.yresolution):
            Y_data_array[j, :] = max_y * j / (self.yresolution-1)
        for i, profile_name in enumerate(profile_paths):
            try:
                prof = md.MesaData(profile_name, read_data=False)
                density_column = next((name for name in ("logRho", "rho")
                                       if name in prof.columns), None)
                prof.read_data(columns + ([density_column] if density_column else []))
            except Exception as e:
                raise RuntimeError(f"Couldn't read profile {profile_name}: {e}") from e
            x_coord = self.function_on_xaxis(prof.header[self._xaxis_column] / xaxis_divide)
            if self.xaxis == "collapse" and x_coord > min_x_coord:
                print("Profiles are not decreasing in X coordinate!!!")
            elif self.xaxis != "collapse" and x_coord < max_x_coord:
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

            if density_column is None:
                self._missing_density_profiles.append(str(profile_name))
            else:
                log_density = prof.get(density_column)[::-1]
                if density_column == "rho":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        log_density = np.log10(log_density)
                self._log_density[:, i] = np.interp(y_interp, y_data, log_density)
                # Do not extend density contours beyond this profile's surface.
                surface = prof.header['star_mass' if self.yaxis == "mass" else 'photosphere_r']
                self._log_density[y_interp > min(surface, y_data.max()), i] = np.nan

        XYZData = namedtuple('XYZData', 'xlims X Y Z')
        return XYZData((min_x_coord, max_x_coord),
                       X_data_array, Y_data_array, Z_data_array)

    def plot(self, *,
             axis=None,
             contour_colormap=None,
             core_masses=None,
             density_contours=None,
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
             save_filename="Kippenhahn.png",
             size=(7, 5)):
        """Draw profile contours, mixing boundaries, and stellar structure.

        Parameters
        ----------
        axis : matplotlib.axes.Axes, optional
            Axes on which to draw. If None, create a figure and axes.
        contour_colormap : matplotlib.colors.Colormap, optional
            Colormap for the filled contours. Defaults to ``'Blues'``
            for most data, and ``'RdBu_r'`` for ``'net_nuclear_energy'``.
        core_masses : sequence of {'He', 'C', 'O', 'CO', 'ONe', 'Fe'}, optional
            Core masses to overlay when ``yaxis='mass'``. Defaults to He, CO, ONe;
            an empty sequence suppresses them.
        density_contours : sequence of float, optional
            Densities in g/cm^3 to overlay as gray lines with inline density
            labels. Values must be finite and positive; they are sorted and
            deduplicated. None or an empty sequence disables the overlay.
            Requires ``logRho`` (log10 density) or ``rho`` in every selected
            profile. Interpolation uses log10 density on the profile grid.
            Each label is placed halfway along the longest visible contour
            for that density, with arc length measured in display coordinates.
        decorate_plot : bool, optional
            Add axis labels and a colorbar.  Default is True.
        levels : array_like, optional
            Explicit, increasing contour levels. If None or empty, generate
            levels from the extracted data. For ``'eps_nuc'`` with automatic
            ``'log'`` levels, the lower bound is set to zero. This corresponds
            to an energy generation rate of one when using logarithmic data;
            the stored data are not clipped or modified.
        level_style : str, optional
            If ``levels`` is not provided, then we can compute the levels
            as "log" (evenly spaced levels in already-logarithmic data), "linear", or
            "symmetric" (centered on zero, and adapting to data range).
            Default is "log" except for ``'net_nuclear_energy'``, which uses
            "symmetric". Ignored when explicit levels are supplied.
        num_levels : int, optional
            Positive number of intervals used by the linear and logarithmic
            level generators. Defaults to 8; ignored for symmetric or explicit
            levels.
        show_conv, show_therm, show_semi, show_over : bool, optional
            Display convection, thermohaline mixing, semiconvection, and
            overshooting boundaries, respectively. Each defaults to True.
        show_mixing_legend : bool, optional
            Include mixing types in the first legend column. Default is True.
        show_core_mass_legend : bool, optional
            Include core-mass lines in the second legend column. Default is True.
        show_plot : bool, optional
            Call ``Figure.show()`` on the figure owning ``axis`` after plotting.
            Default is False; an interactive backend is needed for display.
        show_rot : bool, optional
            Display rotational mixing boundaries. Default is False.
        save_file : bool, optional
            Apply ``Figure.tight_layout()`` and save the figure owning ``axis``
            after the optional show call. Layout is applied before density
            labels are positioned and before the show call. Default is True.
        save_filename : str, optional
            Output filename; its extension determines the format. Defaults to
            ``'Kippenhahn.png'``.
        size : tuple
            Figure size (width, height) in inches.

        Returns
        -------
        KippPlot
            Named tuple containing ``contour_plot`` (the contour set returned
            by ``Axes.contourf``), ``histories`` (loaded ``MesaData`` objects),
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

        Each call reuses ``xyz_data`` and ``mixing_zones`` without rereading
        input files. New artists are added to the supplied axes; existing
        artists are not cleared. Display and file output use the figure owning
        those axes, or the newly created figure when ``axis`` is None.

        """
        return self._plot(
            axis=axis,
            contour_colormap=contour_colormap,
            core_masses=core_masses,
            density_contours=density_contours,
            decorate_plot=decorate_plot,
            levels=levels,
            level_style=level_style,
            num_levels=num_levels,
            show_mixing_legend=show_mixing_legend,
            show_core_mass_legend=show_core_mass_legend,
            show_plot=show_plot,
            show_conv=show_conv,
            show_therm=show_therm,
            show_semi=show_semi,
            show_over=show_over,
            show_rot=show_rot,
            save_file=save_file,
            save_filename=save_filename,
            size=size
        )

    def _plot_history_burning(self, axis, colormap, levels, *, hybrid_mode=False):
        """Render the paths and values prepared by the burning-zone builder."""
        burning_zones = self._build_buring_zones()
        values = burning_zones.burn_types
        if levels is None or len(levels) == 0:
            limit = max(1, int(np.ceil(np.max(np.abs(values))))) if values else 1
            if hybrid_mode:
                finite = self.xyz_data.Z[np.isfinite(self.xyz_data.Z)]
                if finite.size:
                    limit = max(limit, int(np.ceil(np.max(np.abs(finite)))))
            levels = np.arange(-limit - 0.5, limit + 1.5)
        levels = np.asarray(levels, dtype=float)
        if (levels.ndim != 1 or len(levels) < 2 or not np.all(np.isfinite(levels)) or
                np.any(np.diff(levels) <= 0)):
            raise ValueError("levels must contain at least two finite, increasing bin boundaries")
        colormap = plt.get_cmap(colormap)
        norm = BoundaryNorm(levels, colormap.N, clip=True)
        if hybrid_mode:
            axis.contourf(self.xyz_data.X, self.xyz_data.Y, self.xyz_data.Z,
                          cmap=colormap, norm=norm, levels=levels,
                          extend="both", antialiased=False, zorder=0.9)
        collection = PatchCollection([PathPatch(path) for path in burning_zones.zones],
                                     array=np.asarray(values), cmap=colormap,
                                     norm=norm, edgecolors="none", antialiaseds=False,
                                     alpha=1, zorder=1)
        axis.add_collection(collection)
        axis.autoscale_view()
        return collection

    def plot_burning_regions(self, *, hybrid_mode=False, **kwargs):
        """Plot history burning regions with mixing and stellar structure overlays.

        Parameters
        ----------
        hybrid_mode : bool, optional
            Draw profile ``net_nuclear_energy`` underneath opaque history
            regions. Requires construction with ``load_profiles=True`` and
            ``identifier="net_nuclear_energy"``. Default is False. Both layers
            share the colormap and bin boundaries; automatic bins cover both
            data sources. Profile values outside explicit bins use the endpoint
            colors, as do history values.
        **kwargs
            Rendering options accepted by :meth:`plot`. ``levels`` gives bin
            boundaries for the signed logarithmic ``burn_type`` values; the
            default is symmetric integer bins. ``level_style`` and
            ``num_levels`` apply only to profile plots. The default colormap
            is ``RdBu_r``. Density contours still require loaded profiles.

        Returns
        -------
        KippPlot
            Fields ``contour_plot`` (a colored PatchCollection), ``histories``,
            and ``xlims`` (the full history extent).

        Notes
        -----
        Uses ``burn_type_N`` and ``burn_qtop_N`` for mass, or
        ``burn_relr_type_N`` and ``burn_relr_top_N`` for radius. Each region
        spans the previous history coordinate to the current one, with bounds
        from the current sample. Zero and unused (-9999) regions are omitted;
        negative values represent cooling. Adjacent blocks with the same value
        are stitched using the mixing-region algorithm, including splits and
        merges. No vertical rebinning is performed.
        Construct with ``load_profiles=False`` to use history files alone.
        In hybrid mode, omitted history regions reveal the profile contours;
        nonzero history regions take precedence at their finer time resolution.
        """
        if hybrid_mode:
            if self.xyz_data is None:
                raise ValueError("hybrid_mode requires load_profiles=True")
            if self.identifier != "net_nuclear_energy":
                raise ValueError('hybrid_mode requires identifier="net_nuclear_energy"')
        return self._plot(burning_regions=True, hybrid_mode=hybrid_mode, **kwargs)

    def _plot(self, *,
              burning_regions=False,
              hybrid_mode=False,
              axis=None,
              contour_colormap=None,
              core_masses=None,
              density_contours=None,
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
              save_filename="Kippenhahn.png",
              size=(7, 5)):
        """Render either data source with the shared overlays and decorations."""
        xaxis_divide = self._xaxis_divisor

        density_levels = np.asarray([] if density_contours is None else density_contours,
                                    dtype=float)
        if (density_levels.ndim != 1 or not np.all(np.isfinite(density_levels)) or
                np.any(density_levels <= 0)):
            raise ValueError("density_contours must be a sequence of finite, positive densities in g/cm^3")
        density_levels = np.unique(density_levels)
        if density_levels.size and not self.load_profiles:
            raise ValueError("Density contours require load_profiles=True")
        if density_levels.size and self._missing_density_profiles:
            raise ValueError("Density contours require logRho or rho in profile(s): " +
                             ", ".join(self._missing_density_profiles))

        if axis is None:
            fig, axis = plt.subplots()
            fig.set_size_inches(size)
        else:
            fig = axis.figure

        xyz_data = self.xyz_data
        if burning_regions:
            x = self.mixing_zones.x_coords
            if len(x) < 2:
                raise ValueError("At least two history samples are required for burning regions")
            plot_xlims = (float(np.min(x)), float(np.max(x)))
        elif xyz_data is None:
            raise ValueError("plot() requires load_profiles=True")
        else:
            plot_xlims = xyz_data.xlims

        if levels is None:
            levels = []

        if level_style is None:
            level_style = "log"
            if self.identifier == "net_nuclear_energy":
                level_style = "symmetric"

        if core_masses is None:
            core_masses = ["He", "CO", "ONe"]

        if contour_colormap is None:
            if burning_regions or self.identifier == "net_nuclear_energy":
                contour_colormap = plt.get_cmap("RdBu_r")
            else:
                contour_colormap = plt.get_cmap("Blues")

        # only plot if there is data
        if burning_regions:
            contour_plot = self._plot_history_burning(axis, contour_colormap, levels,
                                                     hybrid_mode=hybrid_mode)
        elif xyz_data.Z.size > 0:
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
            elif mixing_zones.mix_types[i] == 2 and show_over:
                label = "Overshooting"
                color = "purple"
                hatch = "x"
                line = 1
            # Semiconvective mixing
            elif mixing_zones.mix_types[i] == 3 and show_semi:
                label = "Semiconvection"
                color = "red"
                hatch = "\\\\"
                line = 1
            # Thermohaline mixing
            elif mixing_zones.mix_types[i] == 4 and show_therm:
                label = "Thermohaline"
                color = "Gold"  # Salmon
                hatch = "||"
                line = 1
            # Rotational mixing
            elif mixing_zones.mix_types[i] == 5 and show_rot:
                label = "Rotational"
                color = "brown"
                hatch = "*"
                line = 1
            # Anonymous mixing
            else:
                label = "_nolegend_"
                color = "white"
                hatch = ""
                line = 0
            patch = PathPatch(zone, fill=False, hatch=hatch,
                              edgecolor=color, alpha=0.8,
                              linewidth=line, label=label)
            axis.add_patch(patch)
            if line:
                mixing_handles.setdefault(mixing_zones.mix_types[i], patch)

        # limit x_coords to data of contours and add line at stellar surface
        for i, x_coord in enumerate(mixing_zones.x_coords):
            if x_coord > plot_xlims[1]:
                break

        # Also fill with white above the plot to cover any remainder of the
        # contour plot
        axis.fill_between(mixing_zones.x_coords, max(mixing_zones.y_coords),
                          mixing_zones.y_coords, color="w")
        surface_end = len(mixing_zones.x_coords) if burning_regions or self.xaxis == "collapse" else i
        axis.plot(mixing_zones.x_coords[:surface_end], mixing_zones.y_coords[:surface_end], "k-")

        density_plot = None
        if density_levels.size:
            log_density = np.ma.masked_invalid(self._log_density)
            if log_density.count():
                log_levels = np.log10(density_levels)
                in_range = (log_levels > log_density.min()) & (log_levels < log_density.max())
                log_levels = log_levels[in_range]
                if log_levels.size:
                    density_plot = axis.contour(
                        self.xyz_data.X, self.xyz_data.Y, log_density,
                        levels=log_levels, colors="0.25", linewidths=1.5,
                        linestyles="solid", corner_mask=False)
                    density_formats = {}
                    for level, density in zip(log_levels, density_levels[in_range]):
                        value = format_scientific(density)
                        density_formats[level] = rf"${value}\,\mathrm{{g/cm^3}}$"

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
                        self.function_on_xaxis(history.get(self._xaxis_column) / xaxis_divide),
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
            cbar = fig.colorbar(contour_plot, ax=axis, pad=0.05)
            if burning_regions:
                cbar.locator = MaxNLocator(nbins=9, integer=True)
                cbar.update_ticks()

            # use pretty labels if we know the field
            label = self.identifier
            if burning_regions or self.identifier == "net_nuclear_energy":
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
            if self.xaxis == "collapse":
                axis.set_xlabel(r'$\log10(t_f - t)$ [' + self.time_units + ']')
            elif self.xaxis == "star_age":
                axis.set_xlabel(r'$t$ ('+self.time_units+')')
            else:
                axis.set_xlabel(r'$Model Number$')
            if self.yaxis == "radius":
                axis.set_ylabel(r'$r/R_\odot$')
            else:
                axis.set_ylabel(r'$m/M_\odot$')

        if burning_regions:
            axis.margins(x=0, y=0)
        if self.xlims is not None:
            axis.set_xlim(self.xlims)
        if self.xaxis == "collapse":
            axis.set_xlim((self.xlims or plot_xlims)[::-1])

        if save_file:
            fig.tight_layout()

        if density_plot is not None:
            # Measure arc lengths only after limits and the colorbar/layout
            # have established the visible axes rectangle.
            fig.canvas.draw()
            positions = []
            for segments in density_plot.allsegs:
                candidates = [_visible_contour_midpoint(segment, axis) for segment in segments]
                position, length = max(candidates, key=lambda candidate: candidate[1],
                                       default=(None, 0))
                if length:
                    positions.append(position)
            if positions:
                # Keep all levels available: some Matplotlib versions index
                # manual labels by contour index, even for a subset of levels.
                axis.clabel(density_plot, fmt=density_formats, inline=True,
                            fontsize="small", manual=positions)

        if show_plot:
            fig.show()
        if save_file:
            fig.savefig(save_filename)

        KippPlot = namedtuple('KippPlot', 'contour_plot histories xlims')

        return KippPlot(contour_plot, mixing_zones.histories, plot_xlims)
