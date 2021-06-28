subroutine caf(f, r, dipole, trans_matrices, box, a, b, c, t0, k, type, energy)
! ==============================================================================
! compute_angle_forces() speedup attempt.
!
! Compile:
!   f2py3 --f90flags="-Ofast" -c compute_angle_forces.f90 -m compute_angle_forces
! Import:
!   from compute_angle_forces import caf as compute_angle_forces__fortran
! ==============================================================================
    use dipole_reconstruction
    implicit none

    real(4), dimension(:,:),     intent(in out) :: f
    real(4), dimension(:,:),     intent(in)     :: r
    real(8), dimension(:,:,:),   intent(in out) :: dipole
    real(8), dimension(:,:,:,:), intent(in out) :: trans_matrices 
    real(8), dimension(:),       intent(in)     :: box
    integer, dimension(:),       intent(in)     :: a
    integer, dimension(:),       intent(in)     :: b
    integer, dimension(:),       intent(in)     :: c
    real(8), dimension(:),       intent(in)     :: t0
    real(8), dimension(:),       intent(in)     :: k
    real(8), dimension(:),       intent(in)     :: type 
    real(8),                    intent(out)     :: energy

    integer :: ind, aa, bb, cc
    real(8), dimension(3) :: ra, rc, ea, ec, fa, fc
    real(8) :: d, ff, xsinph
    real(8) :: xrasin, xrcsin
    real(8) :: cosphi, cosphi2, theta

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
        ff = - k(ind) * d

        xrasin = ff / (norm_a * sinphi)
        xrcsin = ff / (norm_c * sinphi)
        fa = (cosphi * ea - ec) * xrasin
        fc = (cosphi * ec - ea) * xrcsin

        f(aa, :) = f(aa, :) + fa
        f(cc, :) = f(cc, :) + fc
        f(bb, :) = f(bb, :) - (fa + fc)

        energy = energy - 0.5d0 * ff * d
        if (type(ind) == 1.0) then
          call reconstruct(ra, rc, ea, ec, norm_a, norm_c, theta, cosphi, sinphi, r(bb, :), box, dipole(ind, :, :), trans_matrices(ind, :, :, :))
        end if
      end if

    end do
end subroutine
