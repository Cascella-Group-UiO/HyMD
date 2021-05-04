subroutine cdf(f, r, box, a, b, c, d, cm, dm, energy)
! ==============================================================================
! compute_dihedral_forces() speedup attempt.
!
! Compile:
!   f2py3 --f90flags="-Ofast" -c compute_dihedral_forces.f90 -m compute_dihedral_forces
! Import:
!   from compute_dihedral_forces import cdf as compute_dihedral_forces__fortran
! ==============================================================================
  implicit none

  real(4), dimension(:,3), intent(in out) :: force
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
  real(8), dimension(3) :: f, g, h, v, w, sc, force_on_a, force_on_d
  real(8) :: gnorm, vv, ww, fg, hg, df, cosphi, sinphi, phi

  energy = 0.d0
  force = 0.d0
  
  do ind = 1, size(a)
    aa = a(ind) + 1
    bb = b(ind) + 1
    cc = c(ind) + 1
    dd = d(ind) + 1
    
    f = [r(aa,:) - r(bb,:)]
    g = [r(bb,:) - r(cc,:)]
    h = [r(dd,:) - r(cc,:)]
      
    do i = 1, 3
      f(i) = f(i) - box(i) * nint(f(i) / box(i))
      g(i) = g(i) - box(i) * nint(g(i) / box(i))
      h(i) = h(i) - box(i) * nint(h(i) / box(i))
    end do
  
    v = cross(f, g)
    w = cross(h, g)
    vv = dot_product(v, v)
    ww = dot_product(w, w)
    gnorm = sqrt(dot_product(g, g))
    
    cosphi = dot_product(v, w)
    sinphi = dot_product(cross(v, w), g) / gnorm
    phi = atan2(sinphi, cosphi) 

    ! Add check if cosphi > 1 or cosphi < -1?
    ! if (cosphi > 1) cosphi = 1.d0
    ! if (cosphi < -1) cosphi = -1.d0
    
    fg = dot_product(f, g)
    hg = dot_product(h, g)
    
    sc = v * fg / (vv * gn) - w * hg / (ww * gn)
    df = 0.d0

    do i = 0, 7
      energy = energy + cm(i + 1) * (1.d0 + cos(i * phi + dm(i + 1)))
      df = df + m * cm(i + 1) * sin(i * phi + dm(i + 1))
    end do
 
    force_on_a = df * gnorm * v / vv
    force_on_d = df * gnorm * w / ww

    force(aa, :) = force(aa, :) - force_on_a
    force(bb, :) = force(bb, :) + df * sc + force_on_a
    force(cc, :) = force(cc, :) - df * sc - force_on_d
    force(dd, :) = force(dd, :) + force_on_d
  end do

  contains
  function cross(vector1, vector2)
    real, dimension(3) :: cross
    real, dimension(3), intent(in) :: vector1, vector2
    
    cross(1) = vector1(2) * vector2(3) - vector1(3) * vector2(2)
    cross(2) = vector1(3) * vector2(1) - vector1(1) * vector2(3)
    cross(3) = vector1(1) * vector2(2) - vector1(2) * vector2(1)
  end function cross 
end subroutine cdf
