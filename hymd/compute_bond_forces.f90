subroutine cbf(f, r, box, a, b, r0, k, energy)
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
    ! a : (M,) numpy.ndarray
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
    !
    implicit none

    real(4), dimension(:,:), intent(in out) :: f
    real(4), dimension(:,:), intent(in)     :: r
    real(8), dimension(:),   intent(in)     :: box
    integer, dimension(:),   intent(in)     :: a
    integer, dimension(:),   intent(in)     :: b
    real(8), dimension(:),   intent(in)     :: r0
    real(8), dimension(:),   intent(in)     :: k
    real(8),                 intent(out)    :: energy

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

      energy = energy + 0.5d00 * k(ind) * (rab_norm - r0(ind))**2
    end do
end subroutine
