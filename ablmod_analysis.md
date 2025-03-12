# Analysis of ablmod.F90 module
 Surface module :  ABL computation to provide atmospheric data for surface fluxes computation

## Module dependencies
| Library        | Variable |
|----------------|----------|
| abl            |          |
| par_abl        |          |
| phycst         |          |
| dom_oce        | tmask    |
| sbc_oce        | ght_abl  |
|                | ghw_abl  |
|                | e3t_abl  |
|                | e3w_abl  |
|                | jpka     |
|                | jpkam1   |
| sbcblk         |          |
| prtctl         |          |
| iom            |          |
| in_out_manager |          |
| lib_mpp        |          |
| timing         |          |

## abl_step

### Description from ablmod.F90
```
** Purpose :   Time-integration of the ABL model 

** Method  :   Compute atmospheric variables : vertical turbulence 
                             + Coriolis term + newtonian relaxation
                
** Action  : - Advance TKE to time n+1 and compute Avm_abl, Avt_abl, PBLh
             - Advance tracers to time n+1 (Euler backward scheme)
             - Compute Coriolis term with forward-backward scheme (possibly with geostrophic guide)
             - Advance u,v to time n+1 (Euler backward scheme)
             - Apply newtonian relaxation on the dynamics and the tracers
             - Finalize flux computation in psen, pevp, pwndm, ptaui, ptauj, ptaum
```

### Called from
sbc_abl.F90

### Arguments
|        | abl_stp (definition) | sbc_abl (call) | Comment in ablmod.F90 |
| ------ |----------------------|----------------|-----------------------|
| In     | kt                   | kt             | `time step index`     |
|        | psst                 | sst_m          | `sea-surface temperature [Celsius]` |
|        | pssu                 | ssu_m          | `sea-surface u (U-point)` |
|        | pssv                 | ssv_m          | `sea-surface v (V-point)` |
|        | pssq                 | zssq           | `sea-surface humidity` |
|        | pu_dta               | sf(jp_wndi)    | `large-scale windi` |
|        | pv_dta               | sf(jp_wndj)    | `large-scale windj` |
|        | pt_dta               | sf(jp_tair)    | `large-scale pot. temp.` |
|        | pq_dta               | sf(jp_humi)    | `large-scale humidity` |
|        | pslp_dta             | sf(jp_slp)     | `sea-level pressure` |
|        | pgu_dta              | sf(jp_hpgi)    | `large-scale hpgi` |
|        | pgv_dta              | sf(jp_hpgj)    | `large-scale hpgj` |
| In/out | pcd_du               | zcd_du         | `Cd x Du (T-point)` |
|        | psen                 | zsen           | `Ch x Du` |
|        | pevp                 | zevp           | `Ce x Du` |
|        | pwndm                | wndm           | `\|\|uwnd\|\|` |
|        | ptaui                | utau           | `taux` |
|        | ptauj                | vtau           | `tauy` |
|        | ptaum                | taum           | `\|\|tau\|\|` |
| Out    | ptm_su               | tm_su          | `ice-surface temperature [K]` |
|        | pssu_ice             | u_ice          | `ice-surface u (U-point)` |
|        | pssv_ice             | v_ice          | `ice-surface v (V-point)` |
|        | pssq_ice             | zssqi          | `ice-surface humidity` |
|        | pcd_du_ice           | zcd_dui        | `Cd x Du over ice (T-point)` |
|        | psen_ice             | zseni          | `Ch x Du over ice (T-point)` |
|        | pevp_ice             | zevpi          | `Ce x Du over ice (T-point)` |
|        | pwndm_ice            | wndm_ice       | `\|\|uwnd - uice\|\|` |
|        | pfrac_oce            | ato_i          | |
|        | ptaui_ice            | utau_ice       | `ice-surface taux stress (U-point)` |
|        | ptauj_ice            | vtau_ice       | `ice-surface tauy stress (V-point)` |



## abl_zdf_tke

### Description

```
** Purpose :   Time-step Turbulente Kinetic Energy (TKE) equation

** Method  : - source term due to shear
             - source term due to stratification
             - resolution of the TKE equation by inverting
               a tridiagonal linear system

** Action  : - en : now turbulent kinetic energy)
             - avmu, avmv : production of TKE by shear at u and v-points
               (= Kz dz[Ub] * dz[Un] )
```

### Called from
abl_stp subroutine (same module)

### Arguments

None

## smooth_pblh

### Description

```
** Purpose :   2D Hanning filter on atmospheric PBL height
```

### Called from

abl_zdf_tke subroutine (same module)

### Arguments

|        | smooth_pblh (definition) | abl_zdf_tke (call) |
| ------ |--------------------------|--------------------|
| In     | msk                      | msk_abl            |
| In/out | pvar2d                   | pblh               |
