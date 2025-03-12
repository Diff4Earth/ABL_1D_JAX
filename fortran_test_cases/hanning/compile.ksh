#!/bin/bash

source /opt/intel/oneapi/setvars.sh >/dev/null
module load netcdf/netcdf-4.7.2_intel21_hdf5_MPIO

cd /home/alberta/ABL_1D_JAX/fortran_test_cases/hanning

rm *.o
make smooth_pblh


