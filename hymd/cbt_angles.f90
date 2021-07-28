subroutine cbt_angle(a, b, c, box, theta, angle_force)
    ! a, b, c are directly the vectors
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

  real(8), dimension(3), intent(in)  :: a
  real(8), dimension(3), intent(in)  :: b
  real(8), dimension(3), intent(in)  :: c
  real(8), dimension(3), intent(in)  :: box
  real(8),               intent(out) :: phi
  real(8), dimension(3), intent(out) :: fa 
  real(8), dimension(3), intent(out) :: fc
   

  real(8) :: xra, xrc
  real(8) :: cosphi, sinphi, cosphi2
  real(8), dimension(3) :: ra, rc, ea, ec

 ! Don't loop, just calculate the angle from the the triplet given in the dihedral subroutine
  ra = a - b
  ra = ra - box * nint(ra / box)

  rc = c - b
  rc = rc - box * nint(rc / box)

  norm_a = norm2(ra)
  norm_c = norm2(rc)

  ea = ra / norm_a
  ec = rc / norm_c

  cosphi = dot_product(ea, ec)
  cosphi2 = cosphi * cosphi

  if (cosphi2 < 1.0) then
      phi = acos(cosphi)
      sinphi = sin(theta)

      fa = (cosphi * ea - ec) / (norm_a * sinphi)
      fc = (cosphi * ec - ea) / (norm_c * sinphi)
      ! Add dipole reconstruction routine call
      ! call reconstruct(ra, rc, ea, ec, norm_a, norm_c, theta, cosphi, sinphi, b, box, dipole(2 * ind -1: 2 * ind, :), trans_matrices(3 * ind - 2: 3 * ind, :, :))
    end if
end subroutine
