# Hanning test case

We extract the smooth_pblh function from [ablmod.F90 routine](https://forge.nemo-ocean.eu/nemo/nemo/-/blob/main/src/ABL/ablmod.F90?ref_type=heads) and set up a test case to run it on a NANUK12 2D-field and a mask

The [smooth_pblh.F90](smooth_pblh.F90) script reads a given file and the associated mask, applies the smooth_pblh function and writes the smoothed field alongside with the mask (for debugging purpose)

The ERA5 file is made NEMOlike with [script make_ERA5_NEMOlike.ksh](make_ERA5_NEMOlike.ksh)
You compile it by running [compile.ksh](compile.ksh) and run it with [job.ksh](job.ksh)

The output will be a netcdf file called smooth.nc that contains the smoothed variable and the mask
