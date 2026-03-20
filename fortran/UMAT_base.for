subroutine UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD, &
  &  RPL,DDSDDT,DRPLDE,DRPLDT, &
  &  STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME, &
  &  NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT, &
  &  CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC) 

    use iso_c_binding, only: c_int, c_double, c_char, c_null_char

    implicit none

    ! Interface for C function
    interface
      function invoke_pt(ptname, F, props, nprops, psi, cauchy, C66) result(err) bind(C)
        use iso_c_binding, only: c_int, c_double, c_char, c_null_char
        integer(c_int) :: err
        character(c_char), dimension(*) :: ptname
        integer(c_int), value :: nprops
        real(c_double), dimension(nprops) :: props
        real(c_double) :: psi
        real(c_double), dimension(3,3) :: F
        real(c_double), dimension(6) :: cauchy
        real(c_double), dimension(6,6) :: C66
      end function invoke_pt
    end interface

    ! Abaqus UMAT arguments
    real(8) :: SSE, SPD, SCD, RPL, DRPLDT, DTIME, TEMP, DTEMP, CELENT, PNEWDT
    integer :: NTENS, NSTATV, NPROPS, NDI, NSHR, NOEL, NPT, KSPT, KINC, LAYER

    CHARACTER*80 CMNAME
    real(8), DIMENSION(NSTATV) :: STATEV
    real(8), DIMENSION(NTENS,NTENS) :: DDSDDE
    real(8), DIMENSION(NTENS) :: STRESS, DDSDDT, DRPLDE, STRAN, DSTRAN
    real(8), DIMENSION(2) :: TIME
    real(8), DIMENSION(1) :: PREDEF, DPRED
    real(8), DIMENSION(NPROPS) :: PROPS
    real(8), DIMENSION(3) :: COORDS
    real(8), DIMENSION(3,3) :: DROT, DFGRD0, DFGRD1
    real(8), DIMENSION(4) :: JSTEP

    ! Local variables
    integer :: err
    
    err = invoke_pt(trim(CMNAME) // ".pt" // c_null_char, &
                  & DFGRD1, PROPS, NPROPS, & 
                  & SSE, STRESS, DDSDDE)

    ! Check for errors
    if (err .ne. 0) then
      call stdb_abqerr(-3, "ABQnn: invoke_pt error! Code: %I", err, 0, 0)
    end if

    return
end subroutine UMAT
