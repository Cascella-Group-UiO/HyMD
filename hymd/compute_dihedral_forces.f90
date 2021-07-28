subroutine cdf(force, r, box, a, b, c, d, coeff, phase, energy)
  use cbt_angle
  implicit none

  real(4), dimension(:,:), intent(in out) :: force
  real(4), dimension(:,:), intent(in) :: r
  real(8), dimension(:), intent(in) :: box
  integer, dimension(:), intent(in) :: a
  integer, dimension(:), intent(in) :: b
  integer, dimension(:), intent(in) :: c
  integer, dimension(:), intent(in) :: d
  ! integer, dimension(:), intent(in) :: type 
  real(8), dimension(:,:), intent(in) :: coeff_v 
  ! real(8), dimension(:,:), intent(in) :: coeff_k 
  ! real(8), dimension(:,:), intent(in) :: coeff_g 
  real(8), dimension(:,:), intent(in) :: phase_v 
  ! real(8), dimension(:,:), intent(in) :: phase_k
  ! real(8), dimension(:,:), intent(in) :: phase_g

  real(8), intent(out) :: energy
   
  integer :: ind, aa, bb, cc, dd, i
  real(8), dimension(3) :: f, g, h, v, w, sc, force_on_a, force_on_d
  real(8), dimension(5) :: coeff_, phase_
  real(8) :: g_norm, vv, ww, fg, hg
  real(8) ::df_dih, df_ang, cos_phi, sin_phi, phi

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

    c_v = coeff_v(ind, :)
    d_v = phase_v(ind, :)

    df_dih = 0.d0

    do i = 0, 4 
      energy = energy + c_v(i + 1) * (1.d0 + cos(i * phi + d_v(i + 1)))
      df_dih = df_dih + i * c_v(i + 1) * sin(i * phi + d_v(i + 1))
    end do

    if (type == 1) then
      c_k = coeff_k(ind, :)
      d_k = phase_k(ind, :)
      c_g = coeff_g(ind, :)
      d_g = phase_g(ind, :)
      ! V = V_prop + k * (gamma - gamma_0)**2      
      
      ! Levitt-Warshel
      ! gamma_0 = 106 - 13 * cos(phi - 45)
      ! gamma_0 = 1.85d0 - 0.227d0 * cos(phi - 0.785d0)
      ! Not used in the end?
      ! We use another Fourier expansion?

      k = 0.d0
      gamma_0 = 0.d0
      dk = 0.d0
      dg = 0.d0

      do i = 0, 4 
        k = k + c_k(i + 1) * (1.d0 + cos(i * phi + d_k(i + 1)))
        g_0 = g_0 + c_g(i + 1) * (1.d0 + cos(i * phi + d_g(i + 1)))

        dk = dk + i * c_k(i + 1) * sin(i * phi + d_k(i + 1))
        dg = dg + i * c_g(i + 1) * sin(i * phi + d_g(i + 1))
      end do

      call cbt_angle(r(aa,:), r(bb,:), r(cc,:), box, gamm, fa, fc)
      df_ang = -k * (gamm - gamma_0) 

      var_sq = (gamm - gamma_0) * (gamm - gamma_0)
      energy = energy + 0.5d0 * k * var_sq      
      df_dih = df_dih + 0.5d0 * dk * var_sq + df_ang * dg

      fa = df_ang * fa
      fc = df_ang * fc

      force(aa, :) = force(aa, :) + fa
      force(bb, :) = force(bb, :) - fa - fc
      force(cc, :) = force(cc, :) + fc
    end if
 
    sc = v * fg / (vv * g_norm) - w * hg / (ww * g_norm)
    fa = df_dih * g_norm * v / vv
    fd = df_dih * g_norm * w / ww
    
    ! Dihedrals
    force(aa, :) = force(aa, :) - fa
    force(bb, :) = force(bb, :) + df_dih * sc + fa
    force(cc, :) = force(cc, :) - df_dih * sc - fd
    force(dd, :) = force(dd, :) + fd
  end do
end subroutine cdf
