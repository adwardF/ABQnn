subroutine UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD, &
  &  RPL,DDSDDT,DRPLDE,DRPLDT, &
  &  STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME, &
  &  NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT, &
  &  CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC) 


    use iso_c_binding, only: c_int, c_double, c_char, c_null_char

    implicit none

    interface
      function testfunc() result(err) bind(C)
        use iso_c_binding, only: c_int, c_double, c_char, c_null_char
        integer(c_int) :: err
      end function

      function invoke_pt(ptname, F, psi, cauchy, C66) result(err) bind(C)
        use iso_c_binding, only: c_int, c_double, c_char, c_null_char
        integer(c_int) :: err
        character(c_char),dimension(*) :: ptname
        real(c_double) :: psi
        real(c_double),dimension(3,3) :: F
        real(c_double),dimension(6) :: cauchy
        real(c_double),dimension(6,6) :: C66
      end function
    end interface

    real(8) :: SSE,SPD,SCD,RPL,DRPLDT,DTIME,TEMP,DTEMP,CELENT,PNEWDT

    integer :: NTENS, NSTATV, NPROPS, NDI, NSHR, NOEL, NPT,KSPT,KINC,LAYER

    integer :: err

    CHARACTER*80 CMNAME
    real(8), DIMENSION(NSTATV) :: STATEV
    real(8), DIMENSION(NTENS,NTENS) :: DDSDDE
    real(8), DIMENSION(NTENS) :: STRESS, DDSDDT, DRPLDE, STRAN, DSTRAN
    real(8), DIMENSION(2) :: TIME
    real(8), DIMENSION(1) :: PREDEF, DPRED
    real(8), DIMENSION(NPROPS) :: PROPS
    real(8), DIMENSION(3) :: COORDS
    real(8), DIMENSION(3,3) :: DROT,DFGRD0, DFGRD1
    real(8), DIMENSION(4) :: JSTEP

    err = invoke_pt(c_char_"D:/dev/ABQnn/allmodels/"//trim(CMNAME)//".pt"//c_null_char, DFGRD1, SSE, STRESS, DDSDDE)

    if (err .ne. 0) then
      call stdb_abqerr(-3, "invoke pt error ! %I" , err, 0, 0)
    end if

    !call stdb_abqerr(1 , "F %R %R %R; %R %R %R; %R %R %R", 0, DFGRD1, 0)

    !call stdb_abqerr(1 , "STRESS %R %R %R %R %R %R", 0, STRESS, 0)

    !call stdb_abqerr(1 , "DDSDDE 1:6 %R %R %R %R %R %R", 0, DDSDDE, 0)
    
    !call pt_module_invoke

    return
end subroutine