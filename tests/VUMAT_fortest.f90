program VUMAT_fortest

use iso_c_binding, only: c_char, c_null_char, c_double, c_int

implicit none

! Dimensions: nblock=1000, ndir=3, nshr=3
! defgradF: Fortran order (nblock, ndir+2*nshr) = (1000, 9)
! stressNew: Fortran order (nblock, ndir+nshr)  = (1000, 6)
integer, parameter :: NBLOCK = 1000
integer, parameter :: NDIR   = 3
integer, parameter :: NSHR   = 3
integer, parameter :: NDEFGRAD = NDIR + 2 * NSHR   ! 9
integer, parameter :: NSTRESS  = NDIR + NSHR        ! 6

! VUMAT component ordering for defgradF (columns):
!   col 1: F11, col 2: F22, col 3: F33,
!   col 4: F12, col 5: F23, col 6: F31,
!   col 7: F21, col 8: F32, col 9: F13
real(c_double) :: defgradF(NBLOCK, NDEFGRAD)
real(c_double) :: enerInternNew(NBLOCK)
real(c_double) :: stressNew(NBLOCK, NSTRESS)

integer :: i, j
integer(c_int) :: err
integer(c_int) :: nblock_c, ndir_c, nshr_c, n_mat_par

real(c_double) :: mat_par(2)
real(c_double) :: rnd(NDEFGRAD)

interface
    function invoke_pt_vumat_batch( &
            module_name, defgradF, nblock, ndir, nshr, &
            mat_par, n_mat_par, enerInternNew, stressNew) result(err) bind(C)
        use iso_c_binding, only: c_char, c_double, c_int
        integer(c_int) :: err
        character(c_char), dimension(*), intent(in) :: module_name
        real(c_double), dimension(*),    intent(in) :: defgradF
        integer(c_int), value,           intent(in) :: nblock
        integer(c_int), value,           intent(in) :: ndir
        integer(c_int), value,           intent(in) :: nshr
        real(c_double), dimension(*),    intent(in) :: mat_par
        integer(c_int), value,           intent(in) :: n_mat_par
        real(c_double), dimension(*),   intent(out) :: enerInternNew
        real(c_double), dimension(*),   intent(out) :: stressNew
    end function invoke_pt_vumat_batch
end interface

mat_par(1) = 1.0d0
mat_par(2) = 10.0d0
n_mat_par  = 2

nblock_c = NBLOCK
ndir_c   = NDIR
nshr_c   = NSHR

! Generate NBLOCK random deformation gradients in VUMAT component order.
! Layout: defgradF(i, :) = [F11, F22, F33, F12, F23, F31, F21, F32, F13]
do i = 1, NBLOCK
    call random_number(rnd)

    ! Small random perturbation around identity
    rnd = rnd * 0.3d0 - 0.15d0

    ! Direct components: add identity
    defgradF(i, 1) = rnd(1) + 1.0d0  ! F11
    defgradF(i, 2) = rnd(2) + 1.0d0  ! F22
    defgradF(i, 3) = rnd(3) + 1.0d0  ! F33

    ! Off-diagonal components (tensor shear, not engineering)
    defgradF(i, 4) = rnd(4)           ! F12
    defgradF(i, 5) = rnd(5)           ! F23
    defgradF(i, 6) = rnd(6)           ! F31
    defgradF(i, 7) = rnd(7)           ! F21
    defgradF(i, 8) = rnd(8)           ! F32
    defgradF(i, 9) = rnd(9)           ! F13
end do

write(*,*) "Generated", NBLOCK, "random deformation gradients (VUMAT batch)"

enerInternNew = 0.0d0
stressNew     = 0.0d0

! Invoke the batch VUMAT pipeline once for all material points
err = invoke_pt_vumat_batch( &
    c_char_"VUMAT_NH_3D.pt" // c_null_char, &
        defgradF, nblock_c, ndir_c, nshr_c, &
        mat_par, n_mat_par, &
        enerInternNew, stressNew)

if (err /= 0) then
    write(*,*) "Error: invoke_pt_vumat_batch returned code", err
    stop 1
end if

write(*,*) "Batch invocation completed successfully"
write(*,*) "First internal energy:  ", enerInternNew(1)
write(*,*) "Last  internal energy:  ", enerInternNew(NBLOCK)
write(*,*) "First stress (VUMAT order, ndir+nshr=6): ", stressNew(1, :)
write(*,*) "Last  stress:                            ", stressNew(NBLOCK, :)

end program VUMAT_fortest
