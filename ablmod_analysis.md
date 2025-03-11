# Analysis of ablmod.F90 module

## Module dependencies
* abl
* par_abl
* phycst
* dom_oce
  * tmask
* sbc_oce
  * ght_abl
  * ghw_abl
  * e3t_abl
  * e3w_abl
  * jpka
  * jpkam1
* sbcblk
* prtctl
* iom
* in_out_manager
* lib_mpp
* timing

## abl_step

### Called from
sbc_abl.F90

### Arguments
|        | abl_stp (definition) | sbc_abl (call) |
| ------ |----------------------|----------------|
| In     | kt                   | kt             |
|        | psst                 | sst_m          |
|        | pssu                 | ssu_m          |
|        | pssv                 | ssv_m          |
|        | pssq                 | zssq           |
|        | pu_dta               | sf(jp_wndi)    |
|        | pv_dta               | sf(jp_wndj)    |
|        | pt_dta               | sf(jp_tair)    |
|        | pq_dta               | sf(jp_humi)    |
|        | pslp_dta             | sf(jp_slp)     |
|        | pgu_dta              | sf(jp_hpgi)    |
|        | pgv_dta              | sf(jp_hpgj)    |
| In/out | pcd_du               | zcd_du         |
|        | psen                 | zsen           |
|        | pevp                 | zevp           |
|        | pwndm                | wndm           |
|        | ptaui                | utau           |
|        | ptauj                | vtau           |
|        | ptaum                | taum           |
| Out    | ptm_su               | tm_su          |
|        | pssu_ice             | u_ice          |
|        | pssv_ice             | v_ice          |
|        | pssq_ice             | zssqi          |
|        | pcd_du_ice           | zcd_dui        |
|        | psen_ice             | zseni          |
|        | pevp_ice             | zevpi          |
|        | pwndm_ice            | wndm_ice       |
|        | pfrac_oce            | ato_i          |
|        | ptaui_ice            | utau_ice       |
|        | ptauj_ice            | vtau_ice       |



## abl_zdf_tke

### Called from
abl_stp subroutine (same module)

### Arguments

None

## smooth_pblh

### Called from

### Arguments
