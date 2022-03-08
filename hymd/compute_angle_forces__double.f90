subroutine caf_d(f, r, box, a, b, c, t0, k, energy, angle_pr)
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
    real(8), dimension(3),   intent(out)    :: angle_pr

    integer :: ind, aa, bb, cc
    real(8) :: ra_x, ra_y, ra_z, rc_x, rc_y, rc_z
    real(8) :: ea_x, ea_y, ea_z, ec_x, ec_y, ec_z
    real(8) :: fa_x, fa_y, fa_z, fc_x, fc_y, fc_z
    real(8) :: d, ff, bx, by, bz, xsinph, xra, xrc
    real(8) :: xrasin, xrcsin
    real(8) :: cosphi, cosphi2, theta

    energy = 0.0d00
    angle_pr = 0.0d00
    f = 0.0d00

    bx = 1.0d0 / box(1)
    by = 1.0d0 / box(2)
    bz = 1.0d0 / box(3)

    do ind = 1, size(a)
      aa = a(ind) + 1
      bb = b(ind) + 1
      cc = c(ind) + 1

      ra_x = r(aa, 1) - r(bb, 1)
      ra_y = r(aa, 2) - r(bb, 2)
      ra_z = r(aa, 3) - r(bb, 3)
      ra_x = ra_x - box(1) * nint(ra_x * bx)
      ra_y = ra_y - box(2) * nint(ra_y * by)
      ra_z = ra_z - box(3) * nint(ra_z * bz)

      rc_x = r(cc, 1) - r(bb, 1)
      rc_y = r(cc, 2) - r(bb, 2)
      rc_z = r(cc, 3) - r(bb, 3)
      rc_x = rc_x - box(1) * nint(rc_x * bx)
      rc_y = rc_y - box(2) * nint(rc_y * by)
      rc_z = rc_z - box(3) * nint(rc_z * bz)

      xra = 1.0d0 / sqrt(ra_x * ra_x + ra_y * ra_y + ra_z * ra_z)
      xrc = 1.0d0 / sqrt(rc_x * rc_x + rc_y * rc_y + rc_z * rc_z)

      ea_x = ra_x * xra
      ea_y = ra_y * xra
      ea_z = ra_z * xra

      ec_x = rc_x * xrc
      ec_y = rc_y * xrc
      ec_z = rc_z * xrc

      cosphi = ea_x * ec_x + ea_y * ec_y + ea_z * ec_z
      cosphi2 = cosphi * cosphi

      if (cosphi2 < 1.0) then
        theta = acos(cosphi)
        xsinph = 1.0d0 / sqrt(1.0d0 - cosphi2)

        d = theta - t0(ind)
        ff = - k(ind) * d

        xrasin = xra * xsinph * ff
        xrcsin = xrc * xsinph * ff

        fa_x = (ea_x * cosphi - ec_x) * xrasin
        fa_y = (ea_y * cosphi - ec_y) * xrasin
        fa_z = (ea_z * cosphi - ec_z) * xrasin

        fc_x = (ec_x * cosphi - ea_x) * xrcsin
        fc_y = (ec_y * cosphi - ea_y) * xrcsin
        fc_z = (ec_z * cosphi - ea_z) * xrcsin

        f(aa, 1) = f(aa, 1) + fa_x
        f(aa, 2) = f(aa, 2) + fa_y
        f(aa, 3) = f(aa, 3) + fa_z

        f(cc, 1) = f(cc, 1) + fc_x
        f(cc, 2) = f(cc, 2) + fc_y
        f(cc, 3) = f(cc, 3) + fc_z

        f(bb, 1) = f(bb, 1) - (fa_x + fc_x)
        f(bb, 2) = f(bb, 2) - (fa_y + fc_y)
        f(bb, 3) = f(bb, 3) - (fa_z + fc_z)

        energy = energy - 0.5d0 * ff * d

        angle_pr(1) = angle_pr(1) + (fa_x * ra_x) + (fc_x * rc_x)
        angle_pr(2) = angle_pr(2) + (fa_y * ra_y) + (fc_y * rc_y)
        angle_pr(3) = angle_pr(3) + (fa_z * ra_z) + (fc_z * rc_z)

      end if
    end do
end subroutine
