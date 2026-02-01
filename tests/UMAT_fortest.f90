program UMAT_fortest

use iso_c_binding, only: c_char, c_null_char, c_double, c_int

implicit none

real(c_double), dimension(1000,3,3) :: F
real(c_double) :: psi
real(c_double), dimension(6) :: cauchy6
real(c_double), dimension(6,6) :: DDSDDE

integer :: i, j
integer(c_int) :: err

real(c_double) :: mat_par(2)
integer(c_int) :: n_mat_par

interface 
    function invoke_pt(module_name, F, mat_par, n_mat_par, psi, cauchy6, DDSDDE) result(err) bind(C)
        use iso_c_binding, only: c_char, c_null_char, c_double, c_int
        character(c_char), dimension(*), intent(in) :: module_name
        real(c_double), dimension(3,3), intent(in) :: F
        real(c_double), dimension(*), intent(in) :: mat_par
        integer(c_int), intent(in) :: n_mat_par
        real(c_double), intent(out) :: psi
        real(c_double), dimension(6), intent(out) :: cauchy6
        real(c_double), dimension(6,6), intent(out) :: DDSDDE
        integer(c_int) :: err
    end function invoke_pt
end interface

mat_par(1) = 1.0d0
mat_par(2) = 10.0d0
n_mat_par = 2

! Generate random deformation gradients
do i = 1, 1000
    call random_number(F(i,:,:))
    
    ! Scale to realistic deformation range
    F(i,:,:) = (F(i,:,:) * 0.3d0 - 0.15d0)
    
    ! Add identity to make valid deformation gradient
    do j = 1, 3
        F(i,j,j) = F(i,j,j) + 1.0d0
    end do
end do

write(*,*) "Generated 1000 random deformation gradients"

! Test model invocation
do i = 1, 1000
    err = invoke_pt(c_char_"test_NH_3D.pt" // c_null_char, &
                    F(i,:,:), mat_par, n_mat_par, psi, cauchy6, DDSDDE)
    
    if (err /= 0) then
        write(*,*) "Error at iteration", i, ": error code", err
        stop 1
    end if
end do

write(*,*) "All 1000 invocations completed successfully"
write(*,*) "Last psi value:", psi
write(*,*) "Last Cauchy stress:", cauchy6

end program UMAT_fortest
