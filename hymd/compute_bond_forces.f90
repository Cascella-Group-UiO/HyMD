
subroutine cbf(f, r, box, i, j, r0, k, energy)
! ==============================================================================
! compute_bond_forces() speedup attempt.
!
! Compile:
!   f2py3 --f90flags="-Ofast" -c compute_bond_forces.f90 -m compute_bond_forces
! Import:
!   from compute_bond_forces import cbf as compute_bond_forces__fortran
! ==============================================================================
    implicit none

    real(4), dimension(:,:), intent(in out) :: f
    real(4), dimension(:,:), intent(in)     :: r
    real(4), dimension(:),   intent(in)     :: box
    integer, dimension(:),   intent(in)     :: i
    integer, dimension(:),   intent(in)     :: j
    real(4), dimension(:),   intent(in)     :: r0
    real(4), dimension(:),   intent(in)     :: k
    real(4),                 intent(out)    :: energy

    integer :: ind, ii, jj
    real(4) :: rij, rij_x, rij_y, rij_z
    real(4) :: df
    real(4) :: bx, by, bz

    energy = 0.0d00
    f = 0.0d00 ! Set all array elements

    bx = 1.0d00 / box(1)
    by = 1.0d00 / box(2)
    bz = 1.0d00 / box(3)

    do ind = 1, size(i)
      ii = i(ind) + 1
      jj = j(ind) + 1

      rij_x = r(jj, 1) - r(ii, 1)
      rij_x = rij_x - box(1) * nint(rij_x * bx)

      rij_y = r(jj, 2) - r(ii, 2)
      rij_y = rij_y - box(2) * nint(rij_y * by)

      rij_z = r(jj, 3) - r(ii, 3)
      rij_z = rij_z - box(3) * nint(rij_z * bz)

      rij = sqrt(rij_x * rij_x + rij_y * rij_y + rij_z * rij_z)
      df = -k(ind) * (rij - r0(ind))

      f(ii, 1) = f(ii, 1) - df * rij_x / rij
      f(jj, 1) = f(jj, 1) + df * rij_x / rij

      f(ii, 2) = f(ii, 2) - df * rij_y / rij
      f(jj, 2) = f(jj, 2) + df * rij_y / rij

      f(ii, 3) = f(ii, 3) - df * rij_z / rij
      f(jj, 3) = f(jj, 3) + df * rij_z / rij

      energy = energy + 0.5d00 * k(ind) * (rij - r0(ind))**2
    end do
end subroutine
