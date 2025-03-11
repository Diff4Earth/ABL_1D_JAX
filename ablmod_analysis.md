# Analysis of ablmod.F90 module

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

### Called from
abl_stp subroutine (same module)

### Arguments

None

## smooth_pblh

### Called from

abl_zdf_tke subroutine (same module)

### Arguments

|        | smooth_pblh (definition) | abl_zdf_tke (call) |
| ------ |--------------------------|--------------------|
| In     | msk                      | msk_abl            |
| In/out | pvar2d                   | pblh               |
