NETCDF_DIR=/workdir/cryodyn/chekkim/checkelmerice/Intel/install_elmerice/libelmerice/netcdf-4.7.2-install

F90=ifort
FFLAGS=-I$(NETCDF_DIR)/include/ -L$(NETCDF_DIR)/lib -lnetcdf -lnetcdff -O2 -fp-model precise -Dkey_netcdf4 

par_kind.o: par_kind.F90
	$(F90) -c par_kind.F90 $(FFLAGS)

modcdfnames.o: modcdfnames.F90
	$(F90) -c modcdfnames.F90 $(FFLAGS)

cdfio.o: cdfio.F90 modcdfnames.o
	$(F90) -c  cdfio.F90 $(FFLAGS)

smooth_pblh: cdfio.o par_kind.o smooth_pblh.F90
	$(F90) smooth_pblh.F90 -o smooth_pblh cdfio.o modcdfnames.o $(FFLAGS)

