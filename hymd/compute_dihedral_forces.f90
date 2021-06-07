module vector_product
  implicit none
  contains
    function cross(vector1, vector2) result(vector3)
      real(8), dimension(3), intent(in) :: vector1, vector2
      real(8), dimension(3) :: vector3 
      
      vector3(1) = vector1(2) * vector2(3) - vector1(3) * vector2(2)
      vector3(2) = vector1(3) * vector2(1) - vector1(1) * vector2(3)
      vector3(3) = vector1(1) * vector2(2) - vector1(2) * vector2(1)
    end function
end module vector_product

subroutine cdf(force, r, box, a, b, c, d, coeff, phase, energy)
  ! ==============================================================================
  ! compute_dihedral_forces() speedup attempt.
  !
  ! Compile:
  !   f2py3 --f90flags="-Ofast" -c compute_dihedral_forces.f90 -m compute_dihedral_forces
  ! Import:
  !   from compute_dihedral_forces import cdf as compute_dihedral_forces__fortran
  ! ==============================================================================
  use vector_product
  implicit none

  real(4), dimension(:,:), intent(in out) :: force
  real(4), dimension(:,:), intent(in) :: r
  real(8), dimension(:), intent(in) :: box
  integer, dimension(:), intent(in) :: a
  integer, dimension(:), intent(in) :: b
  integer, dimension(:), intent(in) :: c
  integer, dimension(:), intent(in) :: d
  real(8), dimension(:,:), intent(in) :: coeff 
  real(8), dimension(:,:), intent(in) :: phase 
  real(8), intent(out) :: energy
   
  integer :: ind, aa, bb, cc, dd, i
  real(8), dimension(3) :: f, g, h, v, w, sc, force_on_a, force_on_d
  real(8), dimension(5) :: coeff_, phase_
  real(8) :: g_norm, vv, ww, fg, hg, df, cos_phi, sin_phi, phi

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
    g_norm = norm2(g)
    
    cos_phi = dot_product(v, w)
    
    ! Add check if cosphi > 1 or cosphi < -1?
    ! if (cosphi > 1) then
    !   cosphi = 1.d0
    ! if (cosphi < -1) then 
    !   cosphi = -1.d0

    sin_phi = dot_product(cross(v, w), g) / g_norm
    phi = atan2(sin_phi, cos_phi) 

    fg = dot_product(f, g)
    hg = dot_product(h, g)

    coeff_ = coeff(ind, :)
    phase_ = phase(ind, :)
    df = 0.d0

    do i = 0, 4 
      energy = energy + coeff_(i + 1) * (1.d0 + cos(i * phi + phase_(i + 1)))
      df = df + i * coeff_(i + 1) * sin(i * phi + phase_(i + 1))
    end do
 
    sc = v * fg / (vv * g_norm) - w * hg / (ww * g_norm)
    force_on_a = df * g_norm * v / vv
    force_on_d = df * g_norm * w / ww
    
    force(aa,:) = force(aa,:) - force_on_a
    force(bb,:) = force(bb,:) + df * sc + force_on_a
    force(cc,:) = force(cc,:) - df * sc - force_on_d
    force(dd,:) = force(dd,:) + force_on_d
  end do
end subroutine cdf
