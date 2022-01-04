subroutine caf_d(f, r, box, a, b, c, t0, k, energy)
    ! Compute three-particle bond forces and energy
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
    !     Index of particle 1 for M individual three-particle bonds.
    ! b : (M,) numpy.ndarray
    !     Index of particle 2 for M individual three-particle bonds.
    ! c : (M,) numpy.ndarray
    !     Index of particle 3 for M individual three-particle bonds.
    ! t0 : (M,) numpy.ndarray
    !     Equilibrium bond angle for M individual three-particle bonds.
    ! k : (M,) numpy.ndarray
    !     Bond strength for M individual three-particle bonds.
    !
    ! Returns
    ! -------
    ! energy : float
    !     Total energy of all two-particle bonds.
    !
    implicit none

    real(8), dimension(:,:),     intent(in out) :: f
    real(8), dimension(:,:),     intent(in)     :: r
    real(8), dimension(:),       intent(in)     :: box
    integer, dimension(:),       intent(in)     :: a
    integer, dimension(:),       intent(in)     :: b
    integer, dimension(:),       intent(in)     :: c
    real(8), dimension(:),       intent(in)     :: t0
    real(8), dimension(:),       intent(in)     :: k
    real(8),                    intent(out)     :: energy

    integer :: ind, aa, bb, cc
    real(8), dimension(3) :: ra, rc, ea, ec, fa, fc
    real(8) :: d, ff, xsinph, norm_a, norm_c
    real(8) :: xrasin, xrcsin
    real(8) :: cosphi, cosphi2, sinphi, theta

    energy = 0.0d00
    f = 0.0d00

    do ind = 1, size(a)
      aa = a(ind) + 1
      bb = b(ind) + 1
      cc = c(ind) + 1

      ra = r(aa, :) - r(bb, :)
      rc = r(cc, :) - r(bb, :)

      ra = ra - box * nint(ra / box)
      rc = rc - box * nint(rc / box)

      norm_a = norm2(ra)
      norm_c = norm2(rc)
      ea = ra / norm_a
      ec = rc / norm_c

      cosphi = dot_product(ea, ec)
      cosphi2 = cosphi * cosphi

      if (cosphi2 < 1.0) then
        theta = acos(cosphi)
        sinphi = sin(theta)

        d = theta - t0(ind)
        ff = k(ind) * d

        xrasin = -ff / (norm_a * sinphi)
        xrcsin = -ff / (norm_c * sinphi)
        ! 𝜕θ/𝜕cos(θ) * 𝜕cos(θ)/𝜕r
        fa = (ec - cosphi * ea) * xrasin
        fc = (ea - cosphi * ec) * xrcsin

        f(aa, :) = f(aa, :) - fa
        f(cc, :) = f(cc, :) - fc
        f(bb, :) = f(bb, :) + fa + fc

        energy = energy + 0.5d0 * ff * d
      end if
    end do
end subroutine
