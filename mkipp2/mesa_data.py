"""Read selected columns from MESA text output and resolve profile paths."""

from collections import OrderedDict
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from numpy import ma


def _archive_logs_root(archive):
    """Find the unique directory containing MESA index or history files."""
    roots = {str(PurePosixPath(name).parent) for name in archive.namelist()
             if PurePosixPath(name).name in {"profiles.index", "history.data"}}
    if len(roots) != 1:
        raise ValueError(
            f"{archive.filename}: expected one LOGS directory in ZIP archive, "
            f"found {len(roots)}. Specify the directory inside the archive explicitly."
        )
    root = roots.pop()
    return "" if root == "." else root


def resolve_logs_dir(log_dir):
    """Return a directory or a virtual ZIP path to its MESA files.

    Parameters
    ----------
    log_dir : str or path-like
        Local directory, ZIP archive, or explicit directory inside an archive,
        such as ``run.zip/LOGS``.

    Returns
    -------
    str
        Path normalized through ``pathlib.Path``. For an existing ZIP archive
        supplied without an internal directory, append the unique directory
        containing ``profiles.index`` or ``history.data``.

    Raises
    ------
    ValueError
        If an archive supplied without an internal directory contains zero
        or multiple directories with MESA index or history files.

    Notes
    -----
    Ordinary directories and explicit archive-member paths are not checked
    for existence here.
    """
    path = Path(log_dir)
    if path.suffix.lower() == ".zip" and path.is_file():
        with ZipFile(path) as archive:
            root = _archive_logs_root(archive)
        return str(path / root) if root else str(path)
    return str(path)


@contextmanager
def _open_text(filename):
    """Stream a local file or ZIP member, closing all handles on exit."""
    path = Path(filename)
    for parent in path.parents:
        if parent.suffix.lower() == ".zip" and parent.is_file():
            member = path.relative_to(parent).as_posix()
            with ZipFile(parent) as archive:
                if "/" not in member and member not in archive.namelist():
                    member = str(PurePosixPath(_archive_logs_root(archive)) / member)
                try:
                    stream = archive.open(member)
                except KeyError as exc:
                    raise FileNotFoundError(f"{filename}: ZIP member {member!r} not found") from exc
                with stream, TextIOWrapper(stream) as text:
                    yield text
            return
    with open(filename) as text:
        yield text


class MesaData:
    """Read numeric headers and selected columns from a MESA output file.

    Parameters
    ----------
    history_file : str or path-like
        Path to a MESA history or profile text file, including virtual paths
        such as ``run.zip/LOGS/history.data``.
    only_read_header : bool, optional
        Read only the global header, leaving the column mapping empty.
        Takes precedence over ``read_data``. Default is False.
    read_data : bool, optional
        Load column values after reading the header and column mapping.
        If False, defer loading until `read_data` is called. Default is True.
    read_data_cols : list of str, optional
        Columns to load. If None or empty, load all columns. The supplied
        list may be modified to include ``model_number`` when available.
    clean_data : bool, optional
        Remove history rows superseded by later backups or restarts.
        Default is True. Applies only when a ``model_number`` column exists.

    Attributes
    ----------
    filename : str or path-like
        Input file path.
    header : dict of str to float
        Numeric global header values. Known version, compiler, build, SDK,
        date, and math-backend fields are omitted.
    header_num : dict of str to int
        Original zero-based positions of the retained header fields.
    columns : dict of str to int
        Data column names mapped to zero-based file column indices. Empty
        when ``only_read_header`` is True.
    read_columns : list of str
        Unique loaded column names, in requested order. Created when data
        are loaded.
    data : dict of str to numpy.ndarray
        Loaded column arrays, after cleanup and sampling. Created when
        data are loaded.
    num_points : int
        Number of retained data rows. Created when data are loaded.

    Notes
    -----
    The parser expects the standard six-line MESA preamble: header column
    numbers, header names, header values, a blank line, data column numbers,
    and data column names. Numeric rows follow. Header values not explicitly
    skipped must be convertible to float.

    Only selected data columns are parsed. The current array-shape handling
    assumes multiple data rows and does not reliably support single-row or
    empty tables.
    """

    def __init__(self, history_file, *,
                 only_read_header=False, read_data=True, read_data_cols=None,
                 clean_data=True):
        """Read file metadata and optionally load the requested columns."""
        if read_data_cols is None:
            read_data_cols = []

        self.filename = history_file

        # header is a dictionary with the general info from the second
        # and third line of file
        self.header = {}
        self.header_num = {}

        # columns is a dictionary which gives the column number (minus
        # 1) corresponding to the key
        self.columns = {}
        columns = []

        with _open_text(self.filename) as file:

            # first line is not used
            file.readline()

            # following two lines have header data
            header_names = file.readline().split()
            header_vals = file.readline().split()
            i = 0
            for i, header_name in enumerate(header_names):
                # need to properly account for these new columns
                if header_name in ["version_number", "compiler", "build",
                                   "MESA_SDK_version", "date", "math_backend"]:
                    continue
                self.header[header_name] = float(header_vals[i])
                self.header_num[header_name] = i
                i += 1

            if only_read_header:
                return

            # next line is empty
            file.readline()

            # following two lines have column data
            nums = file.readline().split()
            names = file.readline().split()
            for i, name in enumerate(names):
                self.columns[name] = int(nums[i])-1
                columns.append(name)

        if not read_data:
            return

        if len(read_data_cols) == 0:
            read_data_cols = columns
        self.read_data(read_data_cols, clean_data=clean_data)

    def read_data(self, column_names, clean_data=True, sample_every_n=1):
        """Load selected columns, optionally cleaning and sampling history rows.

        Parameters
        ----------
        column_names : list of str
            Nonempty list of columns to load. Duplicates are removed while
            preserving order. If available, ``model_number`` is appended
            to this list in place when not already requested.
        clean_data : bool, optional
            Remove rows superseded by later model numbers. Default is True.
        sample_every_n : int, optional
            Keep every nth row by its one-based position after cleanup,
            always retaining the first and last rows. Default is 1, which
            disables sampling. Sampling requires a ``model_number`` column
            and more than two rows.

        Raises
        ------
        KeyError
            If a requested column is absent from the column mapping.

        Notes
        -----
        The column mapping must already have been read; objects constructed
        with ``only_read_header=True`` do not have that mapping populated.
        Each call rereads the file and replaces ``read_columns``, ``data``,
        and ``num_points`` rather than adding to previously loaded data.

        Cleanup scans backward from the final model, keeping an earlier row
        only when its model number is smaller than every retained later
        model number. This removes both duplicates and obsolete portions
        of a run after a restart. The same row mask is applied to every
        loaded column. Cleanup and sampling are skipped if the file has no
        ``model_number`` data column, as is usual for profiles.
        """
        # always include model_number if its part of the data
        if "model_number" not in column_names and "model_number" in self.columns:
            column_names.append("model_number")

        # be sure there are no repeated column names (could use set
        # but that breaks the order of the columns, which is needed if
        # I want to save the file)
        column_names = list(OrderedDict.fromkeys(column_names))

        self.read_columns = column_names

        # read data
        with _open_text(self.filename) as file:
            data = np.loadtxt(file, skiprows=6,
                              usecols=tuple([self.columns[k] for k in column_names]),
                              unpack=True)

        self.data = {}
        # Be careful in case only one column is required
        if len(data.shape) > 1:
            for i, column in enumerate(column_names):
                self.data[column] = data[i]
        else:
            self.data[column_names[0]] = data

        # clean redos
        if clean_data and "model_number" in self.columns and len(self.data["model_number"]) > 1:
            # create a mask
            model_number = self.data["model_number"]
            mask = np.zeros(len(model_number))
            max_model_number = model_number[-1]

            # last entry is valid, start from there and remove repeats
            for i in range(len(model_number)-2, -1, -1):
                if model_number[i] >= max_model_number:
                    # exclude this point
                    mask[i] = 1
                else:
                    max_model_number = model_number[i]

            if sum(mask) > 0:
                for column in column_names:
                    self.data[column] = ma.masked_array(self.data[column], mask=mask).compressed()

        # subsample points
        if (sample_every_n > 1 and "model_number" in self.columns and
            len(self.data["model_number"]) > 2):
            # keep first and last entry
            # create a mask
            model_number = self.data["model_number"]
            mask = np.zeros(len(model_number))
            for i in range(1, len(model_number)-1):
                if (i+1) % sample_every_n != 0:
                    # exclude this point
                    mask[i] = 1

            if sum(mask) > 0:
                for column in column_names:
                    self.data[column] = ma.masked_array(self.data[column], mask=mask).compressed()

        # count number of points using first entry in dict
        self.num_points = len(self.data[list(self.read_columns)[0]])

    def get(self, key):
        """Return a loaded column by name.

        Parameters
        ----------
        key : str
            Name of a column loaded by `read_data`.

        Returns
        -------
        numpy.ndarray
            Stored column values. The array is returned without copying.

        Raises
        ------
        KeyError
            If the column was not loaded.
        AttributeError
            If data have not yet been loaded.
        """
        return self.data[key]


def get_profile_paths(logs_dirs=None):
    """Resolve profile paths from MESA index files, discarding obsolete rows.

    Parameters
    ----------
    logs_dirs : sequence of str or path-like, optional
        LOGS directories or ZIP archives containing ``profiles.index`` files.
        Explicit paths inside archives are also accepted. Defaults to ``['LOGS']``.

    Returns
    -------
    list of str
        Paths of the form ``<log_dir>/profile<number>.data``, concatenated
        in directory order. Surviving entries retain their index-file order.

    Raises
    ------
    ValueError
        If any index contains no profile entries.

    Notes
    -----
    Read model numbers and profile numbers from the first and third index
    columns, skipping the header line. Scan each index backward, discarding
    rows whose model number is greater than or equal to a retained later
    model number. Cleanup is independent for each directory; entries from
    different directories are neither sorted nor deduplicated together.

    Profile files are not opened or checked for existence. Each index must
    contain at least one data row.
    """
    if logs_dirs is None:
        logs_dirs = ["LOGS"]

    profile_paths = []
    for log_dir in logs_dirs:
        log_dir = resolve_logs_dir(log_dir)
        index_path = log_dir+"/profiles.index"
        with _open_text(index_path) as file:
            model_number, paths = np.loadtxt(file, skiprows=1,
                                             usecols=(0, 2), unpack=True, ndmin=2)
        if paths.size == 0:
            raise ValueError(f"Profile index {index_path} contains no profile entries.")
        mask = np.zeros(len(paths))
        max_model_number = model_number[-1]

        # last entry is valid, start from there and remove repeats
        for i in range(len(model_number)-2, -1, -1):
            if model_number[i] >= max_model_number:
                mask[i] = 1
            else:
                max_model_number = model_number[i]

        if sum(mask) > 0:
            paths = ma.masked_array(paths, mask=mask).compressed()
        profile_paths.extend([log_dir+"/profile"+str(int(i))+".data" for i in paths])
    return profile_paths
