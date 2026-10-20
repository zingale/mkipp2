# Test data

## 15 M_sun star to collapse

The `LOGS-mesa-26.04.1.zip` data is from a 15 M_sun star evolved to
collapse with MESA 26.04.1, and is what is used for the plots
shown in the main docs.

These plots can be generated using `examples.py`.

## Legacy data (not supported)

The `LOGS/` directory is the data that was distributed with the
original `mkipp` package.  The script `examples_old.py` tries to
reproduce the plots that `mkipp` made with this package using the
original data, however, it seems that the integers associated with the
different mixing types have changed in MESA since that original run.
These are defined in
https://github.com/MESAHub/mesa/blob/main/const/public/const_def.f90

This data is kept here for historical sake.


