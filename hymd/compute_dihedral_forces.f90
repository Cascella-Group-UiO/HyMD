subroutine cdf(f, r, box, a, b, c, d, cm, dm, energy)
! ==============================================================================
! compute_dihedral_forces() speedup attempt.
!
! Compile:
!   f2py3 --f90flags="-Ofast" -c compute_dihedral_forces.f90 -m compute_dihedral_forces
! Import:
!   from compute_dihedral_forces import cad as compute_dihedral_forces__fortran
! ==============================================================================
    implicit none

    real(4), dimension(:,:), intent(in out) :: force
!    real(4), dimension(:,:), intent(in) :: r
    real(4), dimension(:,3), intent(in) :: r
    real(8), dimension(3), intent(in) :: box
    integer, dimension(:), intent(in) :: a
    integer, dimension(:), intent(in) :: b
    integer, dimension(:), intent(in) :: c
    integer, dimension(:), intent(in) :: d
    real(8), dimension(:,8), intent(in) :: cm
    real(8), dimension(:,8), intent(in) :: dm
    real(8), intent(out) :: energy
     
    integer :: ind, aa, bb, cc, dd, i
    real(8), dimension(3) :: f, g, h, gv, prj1, prj2, y
    real(8), dimension(3) :: v, w, force_a, force_d
    real(8) :: gnorm, vv, ww, fg, hg, sc, df
    real(8) :: cosphi, sinphi, phi

!    integer :: ind, aa, bb, cc, dd
!    real(8) :: fx, fy, fz
!    real(8) :: gx, gy, gz
!    real(8) :: hx, hy, hz
!    real(8) :: gnx, gny, gnz, gn 
!    real(8) :: vx, vy, vz, vv
!    real(8) :: wx, wy, wx, ww
!    real(8) :: cosphi, sinphi, phi
!    real(8) :: fg, hg, s, df

    ! Why d00 and not just d0? 
    energy = 0.0d00
    force = 0.0d00
    
    ! Why divide 1 / box and not directly the component?
    ! bx = 1.0d0 / box(1)
    ! by = 1.0d0 / box(2)
    ! bz = 1.0d0 / box(3)
    
    do ind = 1, size(a)
      aa = a(ind) + 1
      bb = b(ind) + 1
      cc = c(ind) + 1
      dd = d(ind) + 1
      
!     fx = r(aa, 1) - r(bb, 1)
!     fy = r(aa, 2) - r(bb, 2)
!     fz = r(aa, 3) - r(bb, 3)
!     gx = r(bb, 1) - r(cc, 1)
!     gy = r(bb, 2) - r(cc, 2)
!     gz = r(bb, 3) - r(cc, 3)
!     hx = r(dd, 1) - r(cc, 1)
!     hy = r(dd, 2) - r(cc, 2)
!     hz = r(dd, 3) - r(cc, 3)

!     Will this work??
!     It works!
      f = [r(aa,:) - r(bb,:)]
      g = [r(bb,:) - r(cc,:)]
      h = [r(dd,:) - r(cc,:)]
        
      ! PBC?
      do i = 1, 3
        f(i) = f(i) + box(i) * nint(f(i) / box(i))
        g(i) = g(i) + box(i) * nint(g(i) / box(i))
        h(i) = h(i) + box(i) * nint(h(i) / box(i))
      end do
    
      gnorm = sqrt(dot(g, g))
      gv = g / gn

      prj1 = f - dot_product(f, gv) * gv
      prj2 = h - dot_product(h, gv) * gv
      y = cross(gv, prj1)
      
      cosphi = dot_product(prj1, prj2)
      sinphi = dot_product(y, prj2)
      phi = atan2(sinphi, cosphi) 
      
      v = cross(f, g)
      w = cross(h, g)
      
      vv = dot_product(v, v)
      ww = dot_product(w, w)
      
      fg = dot_product(f, g)
      hg = dot_product(h, g)
      
      sc = a * fg / (vv * gn) - w * hg / (ww * gn)
      df = 0.0d0
      do i = 0, 7
        energy = energy + cm(i + 1) * (1.0d0 + cos(i * phi + dm(i + 1)))
        df = df + m * cm(i + 1) * sin(i * phi + dm(i + 1))
   
      force_a = df * gnorm * v / vv
      force_d = df * gnorm * w / ww

      force(aa, :) = force(aa, :) - force_on_a
      force(bb, :) = force(bb, :) + df * s + force_on_a
      force(cc, :) = force(cc, :) - df * s - force_on_d
      force(dd, :) = force(dd, :) + force_on_d
    end do

    contains
    function cross(a, b)
        real, dimension(3) :: cross
        real, dimension(3), intent(in) :: vector1, vector2
        
        cross(1) = a(2) * b(3) - a(3) * b(2)
        cross(2) = a(3) * b(1) - a(1) * b(3)
        cross(3) = a(1) * b(2) - a(2) * b(1)
    end function cross 
    
    ! Use dot_product function intrinsic in fortran
    ! function dot(a, b)
    !     real :: dot
    !     real, dimension(3), intent(in) :: a, b
    !     
    !     dot = a(1) * b(1) + a(2) * b(2) + a(2) * b(3)
    ! end function dot
end subroutine cdf
