# Analysis of ablmod.F90 module

## abl_step

### Dependencies
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
| In/out |                      |                |
| Out    |                      |                |



## abl_zdf_tke

### Dependencies

### Called from

### Arguments

## smooth_pblh

### Dependencies

### Called from

### Arguments
