NETCDF_DIR=

F90=ifort
FFLAGS=-I$(NETCDF_DIR)/include/ -L$(NETCDF_DIR)/lib -lnetcdf -lnetcdff -O2 -fp-model precise -Dkey_netcdf4 

modcdfnames.o: modcdfnames.F90 modcdfnames_CMIP6.h90
        $(F90) -c modcdfnames.F90 $(FFLAGS)

cdfio.o: cdfio.F90 modcdfnames.o
        $(F90) -c  cdfio.F90 $(FFLAGS)

smooth_pblh: cdfio.o  smooth_pblh.F90
	$(F90) smooth_pblh.F90 -o smooth_pblh cdfio.o modcdfnames.o $(FFLAGS)

