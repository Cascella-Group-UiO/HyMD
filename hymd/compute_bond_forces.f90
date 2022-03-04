
subroutine cbf(f, r, box, i, j, r0, k, energy, bond_pr)
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
    real(8), dimension(3),   intent(out)    :: bond_pr

    integer :: ind, ii, jj
    real(4) :: rij, rij_x, rij_y, rij_z
    real(4) :: df
    real(4) :: bx, by, bz

    energy = 0.0d00
    bond_pr = 0.0d00 !Set x, y, z components to 0
    f = 0.0d00 ! Set all array elements

    bx = 1.0d00 / box(1)
    by = 1.0d00 / box(2)
    bz = 1.0d00 / box(3)

    integer :: ind, aa, bb
    real(8), dimension(3) :: rab, fa
    real(8) :: df, rab_norm

    energy = 0.0d00
    f = 0.0d00

    do ind = 1, size(a)
      aa = a(ind) + 1
      bb = b(ind) + 1

      rab = r(bb, :) - r(aa, :)
      rab = rab - box * nint(rab / box)
      rab_norm = norm2(rab)

      df = k(ind) * (rab_norm - r0(ind))
      fa = -df * rab / rab_norm

      f(aa, :) = f(aa, :) - fa
      f(bb, :) = f(bb, :) + fa

      energy = energy + 0.5d00 * k(ind) * (rij - r0(ind))**2

      bond_pr(1) = bond_pr(1) + ( df * rij_x / rij) * rij_x
      bond_pr(2) = bond_pr(2) + ( df * rij_y / rij) * rij_y
      bond_pr(3) = bond_pr(3) + ( df * rij_z / rij) * rij_z

    end do

end subroutine
