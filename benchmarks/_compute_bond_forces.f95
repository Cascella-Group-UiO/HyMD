
subroutine cbf(f, r, box, i, j, r0, k, energy)
! ==============================================================================
! compute_bond_forces() speedup attempt.
!
! Compile:
!   f2py3 -c _compute_bond_forces.f95 -m _compute_bond_forces
! Import:
!   from _compute_bond_forces import cbf as compute_bond_forces__fortran
! ==============================================================================
    implicit none

    real(8), dimension(:,:), intent(in out) :: f
    real(8), dimension(:,:), intent(in)     :: r
    real(8), dimension(:),   intent(in)     :: box
    integer, dimension(:),   intent(in)     :: i
    integer, dimension(:),   intent(in)     :: j
    real(8), dimension(:),   intent(in)     :: r0
    real(8), dimension(:),   intent(in)     :: k
    real(8),                intent(out)     :: energy

    integer :: ind, ii, jj
    real(8) :: rij, rij_x, rij_y, rij_z, df, bx, by, bz

    energy = 0.0d0
    f = 0.0d0 ! Set all array elements
    bx = 1.0d0 / box(1)
    by = 1.0d0 / box(2)
    bz = 1.0d0 / box(3)

    do ind = 1, size(i)
      ii = i(ind) + 1
      jj = j(ind) + 1

      rij_x = r(ii, 1) - r(jj, 1)
      rij_x = rij_x - bx * nint(rij_x * bx)

      rij_y = r(ii, 2) - r(jj, 2)
      rij_y = rij_y - by * nint(rij_y * by)

      rij_z = r(ii, 3) - r(jj, 3)
      rij_z = rij_z - bz * nint(rij_z * bz)

      rij = sqrt(rij_x * rij_x + rij_y * rij_y + rij_z * rij_z)
      df = k(ind) * (rij - r0(ind))

      f(ii, 1) =  df * rij_x / rij
      f(jj, 1) = -df * rij_x / rij

      f(ii, 2) =  df * rij_y / rij
      f(jj, 2) = -df * rij_y / rij

      f(ii, 3) =  df * rij_z / rij
      f(jj, 3) = -df * rij_z / rij

      energy = energy + 0.5d0 * k(ind) * (rij - r0(ind))**2
    end do
end subroutine
