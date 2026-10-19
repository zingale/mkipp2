# `mkipp2`

Kippenhahn plotter for MESA.

> [!NOTE]
> This is a clean-up of the `mkipp` project.  See `CHANGES.md` for
> history.

> [!IMPORTANT]
> Your `history_columns.list` needs to have `mixing_regions` and
> `mix_relr_regions` specified for MESA to output the necessary data
> of the mixing regions in terms of mass and radius respectively.
> Also, newer versions of MESA include significantly less output in
> the profile files, so values such as `eps_nuc` need to be added to
> `profile_columns.list` as well.
> When combining multiple history files, all files must have the same
> data column names (including unused columns). Column order may differ.


The Kippenhahn plotter consists of 3 files:
- `mesa_data.py`: a simple parser of MESA data files that only parses specific columns to save time
- `kipp_data.py`: construct mixing-region boundaries
- `mkipp.py`: the Kippenhahn plotter

`KippenhahnDiagram` prepares profile and mixing data during construction.
Its `plot()` method produces a decorated plot or draws on a supplied
Matplotlib axes object. The repository comes with some sample MESA
data for a 20 Msun star in `example/LOGS/`. From the repository root,
install the library and run all examples with:

```sh
python -m pip install .
cd example
python example.py
```

The Python snippets below also assume `example/` is the working directory,
since the default input directory is `LOGS/`. For other data locations,
pass `logs_dirs=["path/to/LOGS"]` to `mkipp2.KippenhahnDiagram()`.

ZIP archives can be passed in the same way, without extracting them. For
example, from the repository root:

```python
import mkipp2

k = mkipp2.KippenhahnDiagram(
    logs_dirs=["example/LOGS-mesa-26.04.1.zip"],
    identifier="net_nuclear_energy", log10_on_data=False,
)
k.plot(core_masses=[])
```

Files may be at the archive root or inside a single LOGS folder (including
nested folders). If an archive contains multiple runs, select one explicitly
with `logs_dirs=["runs.zip/path/to/LOGS"]`. Ordinary directories and ZIP paths
can be mixed in the same list. Headers and selected data columns are streamed
from the archive; no temporary files are extracted.

The lower-level readers also accept archive member paths, for example
`MesaData("run.zip/LOGS/history.data")`. `get_profile_paths(["run.zip"])`
returns member paths usable by `MesaData` and `KippenhahnDiagram`.

The seven examples below match `plot1()` through `plot7()` in
[`example/example.py`](example/example.py). Each snippet includes its imports.

The following Python script

```python
import mkipp2

# simple example (saves to default filename Kippenhahn.png)

k = mkipp2.KippenhahnDiagram()
k.plot()
```
results in the following plot
![kipp](images/Kippenhahn.png)

Here is an example that changes the default filename and the x
limits. When x limits are specified the plotter optimizes which
profile data it reads. To save a pdf just change the filename
extension.

```python
import mkipp2

# specify xrange and filename

k = mkipp2.KippenhahnDiagram(xlims=[300, 600])
k.plot(save_filename="Kippenhahn2.png")
```

which results in the following plot
![kipp](images/Kippenhahn2.png)

Several options can be passed to `mkipp2.KippenhahnDiagram()` --- see the docstring
for details. These settings are fixed after construction; create a new
diagram to change them. Rendering options can vary with each call to `plot()`.

Also a matplotlib `Axes` object can be passed to `plot()`. This allows
plot decorations to be made independently, or can be also used to make
a plot with many different Kippenhahn plots. As an example, the
following plots evolution of Helium abundance with respect to time for
the same model as before

```python
import mkipp2
import matplotlib.pyplot as plt
import numpy as np

# plot of Helium abundance against time, independent decoration

fig, axis = plt.subplots()

# k.plot() returns an object containing
#   .contour_plot : the return value of matplotlibs contourf. Can be
#                            used to create a colorbar with plt.colorbar()
#   .histories    : list of history files read. data can be accessed
#                            from this using the get("column_name") function
#   xlims        : limits of data in x coordinate
k = mkipp2.KippenhahnDiagram(xaxis="star_age",
                             time_units="Myr",
                             identifier="y",
                             log10_on_data=False)
plot = k.plot(levels=np.arange(0.0, 1.001, 0.01),
              axis=axis, decorate_plot=False, save_file=False)
cbar = plt.colorbar(plot.contour_plot, pad=0.05)
cbar.set_label("Helium abundance")
axis.set_xlabel("Time (Myr)")
axis.set_ylabel("Mass (solar masses)")
axis.set_xlim(plot.xlims)
fig.tight_layout()
fig.savefig("Kippenhahn3.png")
```

which results in the following plot
![kipp](images/Kippenhahn3.png)

It is also possible to apply a function to the xaxis. This can be used
to generate a log(tf-t) plot. Here we also make use of the `MesaData`
class to read out the maximum age first.

```python
import mkipp2
import matplotlib.pyplot as plt
import numpy as np

# read out max age of star first, then create a log(tf-t) plot

fig, axis = plt.subplots()

# only need to read star_age column first
history = mkipp2.mesa_data.MesaData("LOGS/history.data",
                                    read_data_cols=["star_age"])
max_age = max(history.get("star_age"))
k = mkipp2.KippenhahnDiagram(xaxis="star_age",
                             time_units="yr",
                             function_on_xaxis=lambda x: np.log10(max_age+0.01 - x))
kipp_plot = k.plot(decorate_plot=False, save_file=False, axis=axis)
cbar = plt.colorbar(kipp_plot.contour_plot, pad=0.05)
cbar.set_label("log |eps_nuc|")
axis.set_xlabel("log (tf-t) [yrs]")
axis.set_ylabel("Mass (solar masses)")
fig.tight_layout()
fig.savefig("Kippenhahn4.png")
```

which results in the following plot
![kipp](images/Kippenhahn4.png)

Sometimes you might want to plot data resulting from operations on the
columns of profile files. Instead of re-running a MESA model with
extra columns, to extract the required data you can just pass a
function that computes it. In this case, I make a plot of log g.

```python
import mkipp2
import matplotlib.pyplot as plt
import numpy as np

# add custom extractor to compute g
fig, axis = plt.subplots()
msun = 1.99e33
rsun = 6.96e10
cgrav = 6.67e-8

def g_extractor(identifier, log10_on_data, prof, return_data_columns=False):
    """Return gravitational acceleration, its logarithm, or required columns."""
    if return_data_columns:
        return ["radius", "mass"]
    data = cgrav*prof.get("mass")*msun/(prof.get("radius")*rsun)**2
    if log10_on_data:
        return np.log10(data)
    return data

k = mkipp2.KippenhahnDiagram(extractor=g_extractor)
plot = k.plot(decorate_plot=False, save_file=False, axis=axis)
cbar = plt.colorbar(plot.contour_plot, pad=0.05)
cbar.set_label("log g")
axis.set_xlabel("model_number")
axis.set_ylabel("Mass (solar masses)")
fig.tight_layout()
fig.savefig("Kippenhahn5.png")
```
which results in the following plot
![kipp](images/Kippenhahn5.png)

The prepared data are available through `k.mixing_zones` and `k.xyz_data`
and can be plotted independently without rereading the input files. The following is an
(ugly) example:

```python
import mkipp2
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

# Reading out mixing regions and data, and plotting independently

k = mkipp2.KippenhahnDiagram()

fig, axis = plt.subplots()

# reuse profile data prepared by the constructor
# k.xyz_data contains
#   xyz_data.xlims : limits of data in x coordinate
#   xyz_data.X     : 2D array of horizontal coordinates
#   xyz_data.Y     : 2D array of vertical coordinates (mass or radius)
#   xyz_data.Z     : 2D array of interpolated values from the extractor
# the last three can be used as inputs for matplotlib contour or contourf
xyz_data = k.xyz_data

# reuse mixing regions prepared by the constructor
# k.mixing_zones contains
#   mixing_zones.zones     : matplotlib Path objects for each mixing zone.
#                            These can be plotted using add_patch(...)
#   mixing_zones.mix_types : Integer array containing the type of each zone
#   mixing_zones.x_coords  : x coordinates for points at the surface
#   mixing_zones.y_coords  : y coordinates for points at the surface
#   mixing_zones.histories : mesa_data history files to access additional data
mixing_zones = k.mixing_zones

# just plot convection, overshooting and semiconvection
for i, zone in enumerate(mixing_zones.zones):
    color = ""
    # Convective mixing
    if mixing_zones.mix_types[i] == 1:  # convection
        color = '#332288'
    # Overshooting
    elif mixing_zones.mix_types[i] == 3:  # overshooting
        color = '#117733'
    # Semiconvective mixing
    elif mixing_zones.mix_types[i] == 4:  # semiconvection
        color = '#CC6677'
    else:
        continue
    axis.add_patch(PathPatch(zone, color=color, alpha=0.5, lw=0))

if (xyz_data.Z.size > 0):
    cs = plt.contour(xyz_data.X, xyz_data.Y, xyz_data.Z, [0, 4, 8], colors='k')
    plt.clabel(cs, inline=1, fontsize=10)

axis.plot(mixing_zones.x_coords, mixing_zones.y_coords, 'k', lw=4)
axis.set_xlabel("model_number")
axis.set_ylabel("Mass (solar masses)")
axis.set_xlim(0, max(mixing_zones.x_coords))
axis.set_ylim(0, max(mixing_zones.y_coords))
fig.tight_layout()
fig.savefig("Kippenhahn6.png")
```
which gives the following ugly result (just an example!)
![kipp](images/Kippenhahn6.png)

Another nice thing to do is to make different subpanels containing
different evolutionary stages. This can be achieved by creating a
single `KippenhahnDiagram` object and calling its `plot()` method on various
subaxes, reusing the prepared data

```python
import mkipp2
import matplotlib.pyplot as plt

# create a multi-panel plot

fig = plt.figure(figsize=(9, 5), layout="constrained")

spec = fig.add_gridspec(ncols=4, nrows=1, wspace=0.05, width_ratios=[5, 4, 4, 0.75])
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[0, 2])
ax4 = fig.add_subplot(spec[0, 3])

k = mkipp2.KippenhahnDiagram(xaxis="star_age")
k.plot(axis=ax1, num_levels=10,
       save_file=False, decorate_plot=False)
k.plot(axis=ax2, num_levels=10,
       save_file=False, decorate_plot=False,
       show_mixing_legend=False, show_core_mass_legend=False)
plot = k.plot(axis=ax3, num_levels=10,
              save_file=False, decorate_plot=False,
              show_mixing_legend=False, show_core_mass_legend=False)
cbar = fig.colorbar(plot.contour_plot, cax=ax4)
#cbar.ax.set_yticklabels(['0', '','2','','4','','6','','8',''])
cbar.set_label(r"$\log_{10}|\epsilon_{\rm nuc}|$")

history = mkipp2.mesa_data.MesaData("LOGS/history.data",
                                    read_data_cols=["star_age", "center_h1", "center_he4"])
for i, center_h1 in enumerate(history.get("center_h1")):
    if center_h1 < 1e-3:
        age_tams = history.get("star_age")[i] / 1e6
        break
for i, center_he4 in enumerate(history.get("center_he4")):
    if center_he4 < 1e-3:
        age_hedep = history.get("star_age")[i] / 1e6
        break

ax1.set_xlim([0, age_tams])
ax2.set_xlim([age_tams, age_hedep])
ax3.set_xlim([age_hedep, history.get("star_age")[-1] / 1e6])
ax2.set_xlabel("star age (Myr)")
ax1.set_ylabel("Mass (solar masses)")
ax2.yaxis.set_ticklabels([])
ax3.yaxis.set_ticklabels([])
fig.savefig("Kippenhahn7.png")
```
which produces the following
![kipp](images/Kippenhahn7.png)
