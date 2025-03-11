# Analysis of sbcabl.F90 module

## Module dependencies
| Library        | Variable |
|----------------|----------|
|      abl          |          |
| par_abl | |
|ablmod | |
| ablrst | |
| phycst | |
| fldread| |
| sbc_oce| | 
|sbcblk  | |
|sbc_phy | | 
| dom_oce| tmask |
|iom | |
|in_out_manager | |
|lib_mpp | |
| timing   | |
|lbclnk | |
|prtctl  | |
| si3|u_ice |
| si3|v_ice |
| si3|tm_su |
| si3|ato_i |
| sbc_ice|wndm_ice |
| sbc_ice|utau_ice |
| sbc_ice|vtau_ice |

## Subroutine sbc_abl_init

### Called from
sbcmod.F90 (src/OCE/SBC)

### Purposes : 
- read namelist section namsbc_abl
- initialize and check parameter values
- initialize variables of ABL model

## Subroutine sbc_abl

### Called from
sbcmod.F90 (src/OCE/SBC)

### Purpose
Provide the momentum, heat and freshwater fluxes at the ocean surface from an ABL calculation at each oceanic time step

### Variable: 
kt
