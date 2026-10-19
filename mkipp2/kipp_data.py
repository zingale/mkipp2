"""Read MESA output and construct contour grids and mixing-region boundaries."""

import re
from collections import namedtuple

import numpy as np
from matplotlib.path import Path

import mkipp2.mesa_data as md


class ZoneBlock:
    """A rectangular piece of a mixing region between two history samples.

    The mixing type and vertical bounds at the current sample describe the
    entire interval from the previous x coordinate to the current one. Bounds
    are enclosed mass or radius in solar units.

    Parameters
    ----------
    min_x, max_x : float
        Previous and current x coordinates. Despite their names, they need
        not increase after an axis transform.
    min_y, max_y : float
        Bottom and top of the mixing layer.

    Attributes
    ----------
    vertices : list of ZoneVertex
        Original corners ordered lower left, upper left, upper right,
        lower right.

    Notes
    -----
    ``vertices`` contains four linked corners in the order lower left, upper
    left, upper right, lower right. Initially they form a closed ring. As a
    :class:`Zone` grows, it rewires links between neighboring blocks so that
    following ``next_vertex`` traces the combined boundary instead of each
    rectangle separately. The list retains the original corner objects even
    after their links change.
    """

    def __init__(self, min_x, max_x, min_y, max_y):
        """Create the corners and link them into a circular doubly linked list."""
        # vertices for this block, in the order:
        # -lower left
        # -upper left
        # -upper right
        # -lower right
        self.vertices = [ZoneVertex((min_x, min_y)),
                         ZoneVertex((min_x, max_y)),
                         ZoneVertex((max_x, max_y)),
                         ZoneVertex((max_x, min_y))]
        for i, vertex in enumerate(self.vertices):
            vertex.prev_vertex = self.vertices[i-1]
            vertex.next_vertex = self.vertices[(i+1) % 4]


class ZoneVertex:
    """A boundary node with coordinates, neighbor links, and a traversal flag.

    Parameters
    ----------
    coords : tuple of float
        Boundary coordinates ``(x, y)``.

    Attributes
    ----------
    coords : tuple of float
        Boundary coordinates ``(x, y)``.
    prev_vertex, next_vertex : ZoneVertex or None
        Neighboring boundary nodes; initially unlinked.
    checked : bool
        Whether :meth:`Zone.get_path` has visited the node.
    """

    def __init__(self, coords):
        """Store an ``(x, y)`` pair in an unlinked, unvisited node."""
        self.coords = coords
        self.prev_vertex = None
        self.next_vertex = None
        self.checked = False


class Zone:
    """Accumulate connected blocks of one MESA mixing type into a boundary.

    Parameters
    ----------
    block : ZoneBlock
        Initial block in the current interval.
    mix_type : int
        MESA mixing code for the zone.

    Attributes
    ----------
    old_blocks : list of ZoneBlock
        Archived blocks from earlier intervals.
    last_blocks : list of ZoneBlock
        Blocks from the preceding interval, eligible for extension.
    new_blocks : list of ZoneBlock
        Blocks in the interval being built.
    mix_type : int
        MESA mixing code shared by the zone's blocks.
    open : bool
        Initialized to True; currently unused by the sweep algorithm.

    Notes
    -----
    Blocks are grouped by their position in the sweep through history:
    ``new_blocks`` belong to the interval being built, ``last_blocks`` to the
    preceding interval, and ``old_blocks`` to earlier intervals. Only the
    preceding interval is considered when attaching a new block. Multiple
    blocks can belong to a zone at one interval, allowing regions to split
    and rejoin. Boundary links, rather than a geometric union operation,
    record these connections for conversion to a Matplotlib path.
    """

    def __init__(self, block, mix_type):
        """Start a zone with one new :class:`ZoneBlock` and its mixing code."""
        self.old_blocks = []
        self.last_blocks = []
        self.new_blocks = [block]
        self.mix_type = mix_type
        self.open = True

    def extend(self, upper_vertex, lower_vertex, mix_type):
        """Splice a new block's left edge into matching previous blocks.

        Parameters
        ----------
        upper_vertex : ZoneVertex
            Upper-left corner of the incoming block.
        lower_vertex : ZoneVertex
            Lower-left corner of the incoming block.
        mix_type : int
            MESA mixing code of the incoming block.

        Returns
        -------
        bool
            Whether a previous block of the same mixing type has an
            overlapping or touching vertical interval.

        Notes
        -----
        The caller supplies a block in the adjacent x interval; this method
        checks vertical overlap only. For each match, it connects the old
        upper-right corner to the incoming upper boundary and the incoming
        lower boundary to the old lower-right boundary. This bypasses the
        shared edge while retaining any vertical step between the layers.
        If the incoming layer spans several previous blocks, the lower
        connection is carried forward to splice those boundaries together.

        This mutates vertex links but does not append the block to
        ``new_blocks``. The caller does that and merges any zones bridged by
        the incoming block.
        """
        if mix_type != self.mix_type:
            return False
        found = False
        for block in self.last_blocks:
            old_upper = block.vertices[2]
            old_lower = old_upper.next_vertex
            old_max_y = old_upper.coords[1]
            old_min_y = old_lower.coords[1]
            new_max_y = upper_vertex.coords[1]
            new_min_y = lower_vertex.coords[1]
            if (new_max_y <= old_max_y and new_max_y >= old_min_y) or \
               (new_min_y <= old_max_y and new_min_y >= old_min_y) or \
               (new_max_y > old_max_y and new_min_y < old_min_y):
                # associate vertices
                old_upper.next_vertex = upper_vertex
                upper_vertex.prev_vertex = old_upper
                lower_vertex.next_vertex = old_lower
                old_lower.prev_vertex = lower_vertex
                # if upper vertex is contained in an older block, no need to
                # continue
                if (new_max_y <= old_max_y and new_max_y >= old_min_y) and \
                   (new_min_y <= old_max_y and new_min_y >= old_min_y):
                    return True
                # otherwise, switch lower vertex
                lower_vertex = old_upper
                found = True
        return found

    def switch_new_blocks(self):
        """Advance one interval, archiving old candidates for extension.

        Notes
        -----
        Move ``last_blocks`` into ``old_blocks``, promote ``new_blocks`` to
        ``last_blocks``, and clear ``new_blocks``. An empty ``last_blocks``
        afterward means the zone did not continue into this interval.
        """
        self.old_blocks.extend(self.last_blocks)
        self.last_blocks = self.new_blocks
        self.new_blocks = []

    def get_path(self):
        """Consume the completed boundary links to build a Matplotlib Path.

        Returns
        -------
        matplotlib.path.Path
            Compound path containing the closed boundary loops.

        Notes
        -----
        Include ``last_blocks`` with the archived blocks, then follow each
        unvisited chain of ``next_vertex`` links as a closed subpath. Separate
        loops, including those around holes, share a single compound path.
        The coordinate supplied with ``CLOSEPOLY`` is a placeholder ignored
        by Matplotlib.

        Call only after all blocks have been promoted out of ``new_blocks``.
        This method appends to ``old_blocks`` and marks vertices checked;
        it is a finalization step, not a repeatable accessor.
        """
        self.old_blocks.extend(self.last_blocks)
        vertices = []
        coords = []
        codes = []
        for block in self.old_blocks:
            vertices.extend(block.vertices)

        for starting_vertex in vertices:
            if starting_vertex.checked:
                continue
            starting_vertex.checked = True
            coords.append(starting_vertex.coords)
            codes.append(Path.MOVETO)
            current_vertex = starting_vertex
            while (current_vertex.next_vertex != starting_vertex and
                   not current_vertex.next_vertex.checked):
                current_vertex = current_vertex.next_vertex
                current_vertex.checked = True
                coords.append(current_vertex.coords)
                codes.append(Path.LINETO)
            coords.append((0, 0))
            codes.append(Path.CLOSEPOLY)

        return Path(coords, codes)

    def merge_zone(self, zone2):
        """Collect another zone's blocks after a new block bridges the zones.

        Parameters
        ----------
        zone2 : Zone
            Zone whose blocks are added to this zone.

        Notes
        -----
        The caller must first connect their boundaries with ``extend`` and
        ensure that their mixing types agree. This method only combines the
        three block lists; it neither rewires vertices nor clears ``zone2``.
        The caller removes ``zone2`` from the active zones afterward.
        """
        self.old_blocks.extend(zone2.old_blocks)
        self.last_blocks.extend(zone2.last_blocks)
        self.new_blocks.extend(zone2.new_blocks)
