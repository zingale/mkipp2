# Changes

This is an update of the `mkipp` package, mainly cleaning it up
(pylint, flake8, isort) and making it a proper pip-installable
package.

The HDF5 routines were removed.

LOGS inputs can now be ZIP archives, read directly without extraction.

Density contours can be overlaid as labeled gray lines using
`plot(density_contours=[...])`, with densities supplied in g/cm³.

History burning regions can now be plotted with `plot_burning_regions()`,
including mixing overlays. Use `load_profiles=False` for history-only input.

## History

Based on `SimpleMesaKippy.py` and `Kippy.py`.

Prehistory:

* `SimpleMesaKippy.py`, Author: Alfred Gautschy /23.VIII.2012

* `Kippy.py`, Author: Pablo Marchant /13.V.2013

* `MKippi.py`, Author: Matteo Cantiello 

  * /V.2013 (Added LOSSES. Color/Line Style inspired by convplot.pro IDL routine of A.Heger)
  * /V.2013 (Added RADIUS, TIME and NORMALIZE Options. OMEGA Allows to plot w-contours)
  * /VII.2013 (Tentatively added Rotational diff. coefficients,conv. velocities and Equipartition B-field)

  (most of these were removed and replaced with flexible plotting options)

* `mkipp.py`, Author: Pablo Marchant

  * /II.2014 (full rewrite, code cleanup and now works as a module, not a script)
  * /III.2014 (enhancement of mixing regions plotting to deal with holes and merging regions)
  * /IV.2015 stopped listing enhancements

