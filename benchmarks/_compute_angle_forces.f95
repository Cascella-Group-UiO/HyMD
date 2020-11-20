
subroutine caf(f, r, box, a, b, c, t0, k, energy)
! ==============================================================================
! compute_angle_forces() speedup attempt.
!
! Compile:
!   f2py3 -c _compute_angle_forces.f95 -m _compute_angle_forces
! Import:
!   from _compute_angle_forces import caf as compute_angle_forces__fortran
! ==============================================================================
    implicit none

    real(4), dimension(:,:), intent(in out) :: f
    real(4), dimension(:,:), intent(in)     :: r
    real(8), dimension(:),   intent(in)     :: box
    integer, dimension(:),   intent(in)     :: a
    integer, dimension(:),   intent(in)     :: b
    integer, dimension(:),   intent(in)     :: c
    real(8), dimension(:),   intent(in)     :: t0
    real(8), dimension(:),   intent(in)     :: k
    real(8),                intent(out)     :: energy

    integer :: ind, aa, bb, cc
    real(8) :: ra_x, ra_y, ra_z, rc_x, rc_y, rc_z
    real(8) :: ea_x, ea_y, ea_z, ec_x, ec_y, ec_z
    real(8) :: fa_x, fa_y, fa_z, fc_x, fc_y, fc_z
    real(8) :: d, ff, bx, by, bz, cosphi, theta, xsinph, xra, xrc
    real(8) :: xrasin, xrcsin

    energy = 0.0d0
    f = 0.0 ! Set all array elements

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
      ra_x = ra_x - bx * nint(ra_x * bx)
      ra_y = ra_y - by * nint(ra_y * by)
      ra_z = ra_z - bz * nint(ra_z * bz)

      rc_x = r(cc, 1) - r(bb, 1)
      rc_y = r(cc, 2) - r(bb, 2)
      rc_z = r(cc, 3) - r(bb, 3)
      rc_x = rc_x - bx * nint(rc_x * bx)
      rc_y = rc_y - by * nint(rc_y * by)
      rc_z = rc_z - bz * nint(rc_z * bz)

      xra = 1.0d0 / sqrt(ra_x * ra_x + ra_y * ra_y + ra_z * ra_z)
      xrc = 1.0d0 / sqrt(rc_x * rc_x + rc_y * rc_y + rc_z * rc_z)

      ea_x = ra_x * xra
      ea_y = ra_y * xra
      ea_z = ra_z * xra

      ec_x = rc_x * xrc
      ec_y = rc_y * xrc
      ec_z = rc_z * xrc

      cosphi = ea_x * ec_x + ea_y * ec_y + ea_z * ec_z
      theta = acos(cosphi)
      xsinph = 1.0d0 / sqrt(1.0d0 - cosphi * cosphi)

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

      f(aa, 1) = fa_x
      f(aa, 2) = fa_y
      f(aa, 3) = fa_z

      f(cc, 1) = fc_x
      f(cc, 2) = fc_y
      f(cc, 3) = fc_z

      f(bb, 1) = -(fa_x + fc_x)
      f(bb, 2) = -(fa_y + fc_y)
      f(bb, 3) = -(fa_z + fc_z)

      energy = energy - 0.5d0 * ff * d
    end do
end subroutine
