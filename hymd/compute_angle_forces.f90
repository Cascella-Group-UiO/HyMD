subroutine caf(f, r, box, a, b, c, t0, k, n, energy, angle_pr)
! ==============================================================================
! compute_angle_forces() speedup attempt.
!
! Compile:
!   f2py3 --f90flags="-Ofast" -c compute_angle_forces.f90 -m compute_angle_forces
! Import:
!   from compute_angle_forces import caf as compute_angle_forces__fortran
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
    integer,                 intent(in)     :: n
    real(8),                 intent(out)    :: energy
    real(8), dimension(n,3),   intent(out)    :: angle_pr

    real(8), dimension(3) :: faa, fcc, ra, rc
    integer :: ind, d_, aa, bb, cc
    real(8) :: ea_x, ea_y, ea_z, ec_x, ec_y, ec_z
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

      ra(1) = r(aa, 1) - r(bb, 1)
      ra(2) = r(aa, 2) - r(bb, 2)
      ra(3) = r(aa, 3) - r(bb, 3)
      ra(1) = ra(1) - box(1) * nint(ra(1) * bx)
      ra(2) = ra(2) - box(2) * nint(ra(2) * by)
      ra(3) = ra(3) - box(3) * nint(ra(3) * bz)

      rc(1) = r(cc, 1) - r(bb, 1)
      rc(2) = r(cc, 2) - r(bb, 2)
      rc(3) = r(cc, 3) - r(bb, 3)
      rc(1) = rc(1) - box(1) * nint(rc(1) * bx)
      rc(2) = rc(2) - box(2) * nint(rc(2) * by)
      rc(3) = rc(3) - box(3) * nint(rc(3) * bz)

      xra = 1.0d0 / sqrt(ra(1) * ra(1) + ra(2) * ra(2) + ra(3) * ra(3))
      xrc = 1.0d0 / sqrt(rc(1) * rc(1) + rc(2) * rc(2) + rc(3) * rc(3))

      ea_x = ra(1) * xra
      ea_y = ra(2) * xra
      ea_z = ra(3) * xra

      ec_x = rc(1) * xrc
      ec_y = rc(2) * xrc
      ec_z = rc(3) * xrc

      cosphi = ea_x * ec_x + ea_y * ec_y + ea_z * ec_z
      cosphi2 = cosphi * cosphi

      if (cosphi2 < 1.0) then
        theta = acos(cosphi)

        xsinph = 1.0d0 / sqrt(1.0d0 - cosphi2)

        d = theta - t0(ind)
        ff = - k(ind) * d

        xrasin = xra * xsinph * ff
        xrcsin = xrc * xsinph * ff

        faa(1) = (ea_x * cosphi - ec_x) * xrasin
        faa(2) = (ea_y * cosphi - ec_y) * xrasin
        faa(3) = (ea_z * cosphi - ec_z) * xrasin

        fcc(1) = (ec_x * cosphi - ea_x) * xrcsin
        fcc(2) = (ec_y * cosphi - ea_y) * xrcsin
        fcc(3) = (ec_z * cosphi - ea_z) * xrcsin

        f(aa, 1) = f(aa, 1) + faa(1)
        f(aa, 2) = f(aa, 2) + faa(2)
        f(aa, 3) = f(aa, 3) + faa(3)

        f(cc, 1) = f(cc, 1) + fcc(1)
        f(cc, 2) = f(cc, 2) + fcc(2)
        f(cc, 3) = f(cc, 3) + fcc(3)

        f(bb, 1) = f(bb, 1) - (faa(1) + fcc(1))
        f(bb, 2) = f(bb, 2) - (faa(2) + fcc(2))
        f(bb, 3) = f(bb, 3) - (faa(3) + fcc(3))

        energy = energy - 0.5d0 * ff * d

        do d_ = 1, 3
          angle_pr(aa, d_) = angle_pr(aa, d_) + ( faa(d_) * ra(d_) + fcc(d_) * rc(d_) )/3
          angle_pr(cc, d_) = angle_pr(cc, d_) + ( faa(d_) * ra(d_) + fcc(d_) * rc(d_) )/3
          angle_pr(bb, d_) = angle_pr(bb, d_) + ( faa(d_) * ra(d_) + fcc(d_) * rc(d_) )/3
        end do

      end if
    end do
end subroutine
