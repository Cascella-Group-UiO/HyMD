subroutine cdf(force, r, box, a, b, c, d, coeff, phase, energy)
  use dipole_reconstruction
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

    f = [r(aa, :) - r(bb, :)]
    g = [r(bb, :) - r(cc, :)]
    h = [r(dd, :) - r(cc, :)]
      
    f = f - box * nint(f / box)
    g = g - box * nint(g / box)
    h = h - box * nint(h / box)
  
    v = cross(f, g)
    w = cross(h, g)
    vv = dot_product(v, v)
    ww = dot_product(w, w)
    g_norm = norm2(g)
    
    cos_phi = dot_product(v, w)

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
    
    force(aa, :) = force(aa, :) - force_on_a
    force(bb, :) = force(bb, :) + df * sc + force_on_a
    force(cc, :) = force(cc, :) - df * sc - force_on_d
    force(dd, :) = force(dd, :) + force_on_d
  end do
end subroutine cdf
