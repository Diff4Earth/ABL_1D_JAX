#!/bin/bash

cp ERA5_blh.nc ERA5_blh_NEMOlike.nc

ncrename -d valid_time,time_counter ERA5_blh_NEMOlike.nc
ncrename -d longitude,x ERA5_blh_NEMOlike.nc
ncrename -d latitude,y ERA5_blh_NEMOlike.nc
