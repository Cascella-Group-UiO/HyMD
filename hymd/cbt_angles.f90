subroutine cbt_angle(a, b, c, box, theta)
    ! a, b, c are directly the vectors
! ==============================================================================
! compute_angle_forces() speedup attempt.
!
! Compile:
!   f2py3 --f90flags="-Ofast" -c compute_angle_forces.f90 -m compute_angle_forces
! Import:
!   from compute_angle_forces import caf as compute_angle_forces__fortran
! ==============================================================================
    implicit none

    real(8), dimension(3),   intent(in)     :: a
    real(8), dimension(3),   intent(in)     :: b
    real(8), dimension(3),   intent(in)     :: c
    real(8), dimension(3),   intent(in)     :: box
    real(8),                intent(out)     :: theta

    real(8) :: xra, xrc
    real(8) :: cosphi, cosphi2
    real(8), dimension(3) :: ra, rc, ea, ec

   ! Don't loop, just calculate the angle from the the triplet given in the dihedral subroutine
    ra = a - b
    ra = ra - box * nint(ra / box)

    rc = c - b
    rc = rc - box * nint(rc / box)

    xra = 1.0d0 / sqrt(dot_product(ra, ra))
    xrc = 1.0d0 / sqrt(dot_product(rc, rc))

    ea = ra * xra
    ec = rc * xrc

    cosphi = dot_product(ea, ec)
    cosphi2 = cosphi * cosphi

    if (cosphi2 < 1.0) then
        theta = acos(cosphi)
        ! Add dipole reconstruction routine call
    end if
end subroutine
