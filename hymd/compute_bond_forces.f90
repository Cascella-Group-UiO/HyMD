subroutine cbf(f, r, box, i, j, r0, k, energy, bond_pr)
    ! Compute two-particle bond forces and energy
    !
    ! Parameters
    ! ---------
    ! f : (N,D) numpy.ndarray
    !     Forces for N particles in D dimensions. Changed in place.
    ! r : (N,D) numpy.ndarray
    !     Positions for N particles in D dimensions.
    ! box : (D,) numpy.ndarray
    !     D-dimensional simulation box size.
    ! a : (M,) numpys.ndarray
    !     Index of particle 1 for M individual two-particle bonds.
    ! b : (M,) numpy.ndarray
    !     Index of particle 2 for M individual two-particle bonds.
    ! r0 : (M,) numpy.ndarray
    !     Equilibrium bond distance for M individual two-particle bonds.
    ! k : (M,) numpy.ndarray
    !     Bond strength for M individual two-particle bonds.
    !
    ! Returns
    ! -------
    ! energy : float
    !     Total energy of all two-particle bonds.
    ! bond_pr : (3,) numpy.ndarray
    !     Total bond pressure due all two-particle bonds.
    !
    implicit none

    real(4), dimension(:,:), intent(in out) :: f
    real(4), dimension(:,:), intent(in)     :: r
    real(8), dimension(:),   intent(in)     :: box
    integer, dimension(:),   intent(in)     :: i
    integer, dimension(:),   intent(in)     :: j
    real(8), dimension(:),   intent(in)     :: r0
    real(8), dimension(:),   intent(in)     :: k
    real(8),                 intent(out)    :: energy
    real(4), dimension(3),   intent(out)    :: bond_pr

    integer :: ind, ii, jj
    real(8) :: rij, rij_x, rij_y, rij_z
    real(8) :: df
    real(8) :: bx, by, bz

    energy = 0.0d00
    bond_pr = 0.0d00 !Set x, y, z components to 0
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

      bond_pr(1) = bond_pr(1) + ( df * rij_x / rij) * rij_x
      bond_pr(2) = bond_pr(2) + ( df * rij_y / rij) * rij_y
      bond_pr(3) = bond_pr(3) + ( df * rij_z / rij) * rij_z

    end do

end subroutine
