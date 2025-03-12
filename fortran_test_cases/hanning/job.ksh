#!/bin/bash
#SBATCH -J smooth

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=meom

#SBATCH --mem=4000

#SBATCH --time=01:00:00
#SBATCH --output smooth.%j.output
#SBATCH --error  smooth.%j.error

ulimit -s unlimited

source /opt/intel/oneapi/setvars.sh >/dev/null
module load netcdf/netcdf-4.7.2_intel21_hdf5_MPIO

cd /home/alberta/ABL_1D_JAX/fortran_test_cases/hanning

./smooth_pblh -t ERA5_blh_NEMOlike.nc -v blh -m ERA5_blh_NEMOlike.nc -vm lsm


