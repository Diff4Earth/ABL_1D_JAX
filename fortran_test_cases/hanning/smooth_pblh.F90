PROGRAM smooth_2Dvar
  !!======================================================================
  !!                       *** smooth_2Dvar
  !! This program applies a hanning filter on a 2D field
  !!======================================================================

  IMPLICIT NONE

  INTEGER(KIND=4)                           :: narg, iargc, ijarg  
  CHARACTER(LEN=256)                        :: cf_tfil , cf_mfil  
  CHARACTER(LEN=256)                        :: cn_var, cn_varm
  INTEGER(KIND=4)                           :: npiglo, npjglo  ! size of the domain
  INTEGER(KIND=4)                           ::  npt        ! size of the domain
  CHARACTER(LEN=256)                        :: clunits            ! attribute of output file : units
  CHARACTER(LEN=256)                        :: cldum              ! dummy char variable
  CHARACTER(LEN=256)                        :: cllong_name        !     "      long name
  CHARACTER(LEN=256)                        :: cglobal            !     "      global 
  CHARACTER(LEN=256)                        :: clshort_name       !     "      short name
  INTEGER(KIND=4)                           :: ncout           ! ncid of output file
  INTEGER(KIND=4), ALLOCATABLE, DIMENSION(:):: id_varout  
  REAL(KIND=4), DIMENSION(:,:), ALLOCATABLE :: zmask
  REAL(KIND=4), DIMENSION(:,:), ALLOCATABLE :: zv
  CHARACTER(LEN=256) :: cn_x='x'               !: longitude, I dimension
  CHARACTER(LEN=256) :: cn_y='y'               !: latitude,  J dimension
  CHARACTER(LEN=256) :: cn_t='time_counter'    !: time dimension
  CHARACTER(LEN=256) :: cf_out='smooth.nc' ! output file name

  USE cdfio
  USE par_kind

  narg= iargc()
  IF ( narg == 0 ) THEN
     PRINT *,' usage : smooth_2Dvar  -t T-file -v 2Dvar -m mask-file -vm mask-var'
     PRINT *,'      '
     PRINT *,'     ARGUMENTS :'
     PRINT *,'       -t T-file   : netcdf file that contains the 2D field to be smoothed'
     PRINT *,'       -m mask-file   : netcdf file that contains the 2D mask'
     PRINT *,'       -v 2Dvar   : name of the variable to be smoothed'
     PRINT *,'       -vm mask-var   : name of the mask'
  ENDIF


  ijarg   = 1

  DO   WHILE ( ijarg <= narg )
     CALL getarg (ijarg, cldum) ; ijarg = ijarg + 1
     SELECT CASE ( cldum )
     CASE ( '-t'      ) ; CALL getarg (ijarg, cf_tfil) ; ijarg = ijarg + 1
     CASE ( '-m'      ) ; CALL getarg (ijarg, cf_mfil) ; ijarg = ijarg + 1
     CASE ( '-v'      ) ; CALL getarg (ijarg, cn_var) ; ijarg = ijarg + 1
     CASE ( '-vm'      ) ; CALL getarg (ijarg, cn_varm) ; ijarg = ijarg + 1
     CASE DEFAULT    ; PRINT *,' ERROR : ',TRIM(cldum),' : unknown option.' ; STOP 99
     END SELECT
  ENDDO



  npiglo = getdim (cf_tfil,cn_x)
  npjglo = getdim (cf_tfil,cn_y)
  npt    = getdim (cf_tfil,cn_t)

  PRINT *, 'npiglo =', npiglo
  PRINT *, 'npjglo =', npjglo
  PRINT *, 'npt    =', npt

  CALL CreateOutput

  ! Allocate arrays
  ALLOCATE (zmask(npiglo,npjglo) )
  ALLOCATE (zv(npiglo,npjglo) )

  CALL smooth_pblh( zv, zmask )

  ierr = putvar(ncout , id_varout(1), zv , 0, jpiglo, jpjglo )

  ierr = closeout(ncout )

CONTAINS

  SUBROUTINE CreateOutput
    !!---------------------------------------------------------------------
    !!                  ***  ROUTINE CreateOutput  ***
    !!
    !! ** Purpose :  Set up all things required for the output file, create
    !!               the file and write the header part.
    !!
    !! ** Method  :  Use global module variables
    !!
    !!----------------------------------------------------------------------
    INTEGER(KIND=4) :: jv   ! dummy loop index
    !!----------------------------------------------------------------------
    ! prepare output variables
    ALLOCATE (id_varout(1), stypvar(1) )
    ierr=getvaratt (cf_tfil  , cn_var, clunits, zspval, cllong_name, clshort_name)

    ! define new variables for output 
    stypvar(1)%rmissing_value    = 99999.
    stypvar(1)%scale_factor      = 1.
    stypvar(1)%add_offset        = 0.
    stypvar(1)%savelog10         = 0.
    stypvar(1)%conline_operation = 'N/A'

    stypvar(1)%cunits         = TRIM(clunits)
    stypvar(1)%valid_min      = -100000.
    stypvar(1)%valid_max      =  100000.
    stypvar(1)%cname          = 'degraded_'//TRIM(cv_nam)
    stypvar(1)%clong_name     = 'degraded '//TRIM(cllong_name)
    stypvar(1)%cshort_name    = 'degraded_'//TRIM(clshort_name)

    ncout = create      (cf_out, cf_tfil, npiglo,    npjglo, 0,          ld_nc4=.False. )
    ierr  = createvar   (ncout,  stypvar, 1, 1,    id_varout , ld_nc4=.False. )
    ierr  = putheadervar(ncout,  cf_tfil, npiglo,    npjglo, 0 )
  END SUBROUTINE CreateOutput

!===================================================================================================
   SUBROUTINE smooth_pblh( pvar2d, msk )
!---------------------------------------------------------------------------------------------------

      !!----------------------------------------------------------------------
      !!                   ***  ROUTINE smooth_pblh  ***
      !!
      !! ** Purpose :   2D Hanning filter on atmospheric PBL height
      !!
      !! ---------------------------------------------------------------------

      REAL(wp), DIMENSION(A2D(1)), INTENT(in   ) :: msk
      REAL(wp), DIMENSION(A2D(1)), INTENT(inout) :: pvar2d
      INTEGER                                    :: ji,jj
      REAL(wp)                                   :: smth_a, smth_b
      REAL(wp), DIMENSION(A2D(1))                :: zdX,zdY,zFX,zFY
      REAL(wp)                                   :: zumsk,zvmsk

      !!=========================================================
      !!
      !! Hanning filter
      smth_a = 1._wp / 8._wp
      smth_b = 1._wp / 4._wp
      !
      DO jj = 2, npjglo - 1; DO ji = 2, npiglo
         zumsk = msk(ji,jj) * msk(ji+1,jj)
         zdX ( ji, jj ) = ( pvar2d( ji+1,jj ) - pvar2d( ji  ,jj ) ) * zumsk
      END DO   ;   END DO

      DO jj = 2, npjglo; DO ji = 2, npiglo - 1
         zvmsk = msk(ji,jj) * msk(ji,jj+1)
         zdY ( ji, jj ) = ( pvar2d( ji, jj+1 ) - pvar2d( ji  ,jj ) ) * zvmsk
      END DO   ;   END DO

      DO jj = 2, npjglo; DO ji = 1, npiglo
         zFY ( ji, jj  ) =   zdY ( ji, jj   )                        &
            & +  smth_a*  ( (zdX ( ji, jj+1 ) - zdX( ji-1, jj+1 ))   &   ! add () for NP repro
            &            -  (zdX ( ji, jj   ) - zdX( ji-1, jj   ))  )
      END DO   ;   END DO

      DO jj = 1, npjglo; DO ji = 2, npiglo
         zFX( ji, jj  ) =    zdX( ji, jj   )                         &
           &    + smth_a*(  (zdY( ji+1, jj ) - zdY( ji+1, jj-1))     &   ! add () for NP repro
           &             -  (zdY( ji  , jj ) - zdY( ji  , jj-1)) )
      END DO   ;   END DO

      DO jj = 1, npjglo; DO ji = 1, npiglo
         pvar2d( ji  ,jj ) = pvar2d( ji  ,jj )              &
  &         + msk(ji,jj) * smth_b * (                       &
  &                   ( zFX( ji, jj ) - zFX( ji-1, jj ) )   &   ! add () for NP repro
  &                 + ( zFY( ji, jj ) - zFY( ji, jj-1 ) ) )
      END DO   ;   END DO

!---------------------------------------------------------------------------------------------------
   END SUBROUTINE smooth_pblh
!===================================================================================================

END PROGRAM
