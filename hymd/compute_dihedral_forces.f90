subroutine cdf(force, r, dipoles, trans_matrix, box, a, b, c, d, coeff, dtype, bb_index, dipole_flag, energy)
  use dipole_reconstruction
  implicit none

  real(4), dimension(:,:), intent(in out) :: force
  real(4), dimension(:,:), intent(in) :: r
  real(4), dimension(:,:,:), intent(in out) :: dipoles
  real(4), dimension(:,:,:,:), intent(in out) :: trans_matrix
  real(8), dimension(:), intent(in) :: box
  integer, dimension(:), intent(in) :: a
  integer, dimension(:), intent(in) :: b
  integer, dimension(:), intent(in) :: c
  integer, dimension(:), intent(in) :: d
  real(4), dimension(:,:,:), intent(in) :: coeff 
  integer, dimension(:), intent(in) :: dtype
  integer, dimension(:), intent(in) :: bb_index
  integer, intent(in) :: dipole_flag
  real(8), intent(out) :: energy
   
  integer :: ind, aa, bb, cc, dd, i
  real(8), dimension(3) :: f, g, h, v, w, sc, fa, fb, fc, fd
  real(8), dimension(5) :: c_v, c_k, d_v, d_k
  ! real(8), dimension(5) :: c_g, d_g
  real(8) :: energy_cbt, df_cbt
  real(8) :: g_norm, v_sq, w_sq, f_dot_g, h_dot_g
  real(8) :: df_dih, df_ang, cos_phi, sin_phi, phi

  energy = 0.d0
  df_dih = 0.d0
  force = 0.d0
  dipoles = 0.d0
  trans_matrix = 0.d0

  do ind = 1, size(a)
    aa = a(ind) + 1
    bb = b(ind) + 1
    cc = c(ind) + 1
    dd = d(ind) + 1

    f = r(aa, :) - r(bb, :)
    g = r(bb, :) - r(cc, :)
    h = r(dd, :) - r(cc, :)
      
    f = f - box * nint(f / box)
    g = g - box * nint(g / box)
    h = h - box * nint(h / box)

  
    v = cross(f, g)
    w = cross(h, g)
    v_sq = dot_product(v, v)
    w_sq = dot_product(w, w)
    g_norm = norm2(g)
    
    cos_phi = dot_product(v, w)

    sin_phi = dot_product(cross(v, w), g) / g_norm
    phi = -1.d0 * atan2(sin_phi, cos_phi) 

    ! Improper dihedrals
    ! if (dtype == 2) then
    ! 
    ! end if

    f_dot_g = dot_product(f, g)
    h_dot_g = dot_product(h, g)

    ! Get shape of coeff to see how many arrays we have 
    ! and use the shape to select the arrays
    c_shape = shape(coeff)
    c_v = coeff(ind, 1, :)
    d_v = coeff(ind, 2, :)
    call cosine_series(c_v, c_v, energy, gradient)

    if (c_shape(1) > 4) then
      ! c_shape = (8, 5)
      call cosine_series(coeff(ind, 3, :), coeff(ind, 4, :), energy, gradient)
      call cosine_series(coeff(ind, 5, :), coeff(ind, 6, :), energy, gradient)
    end if

    if (dtype(ind) == 1) then
      ! CBT potential
      ! V = V_prop + k * (gamma - gamma_0)**2      

      c_k = coeff(ind, (c_shape(1) - 1), :)
      d_k = coeff(ind, c_shape(1), :)
      ! These are needed if gamma_0 is expressed as cosine series, not implemented
      ! c_g = coeff(ind, 5, :)
      ! d_g = phase(ind, 6, :)

      call reconstruct( &
        f, r(bb, :), -g, box, c_k, d_k, phi, dipole_flag, &
        energy_cbt, df_cbt, fa, fb, fc, dipoles(ind, 1:2, :), trans_matrix(ind, 1:3, :, :))

      energy = energy + energy_cbt
      df_dih = df_dih + df_cbt

      ! Angle forces
      force(aa, :) = force(aa, :) - fa
      force(bb, :) = force(bb, :) - fb
      force(cc, :) = force(cc, :) - fc

      if (bb_index(ind) == 1) then
        ! calculate last angle in the chain
        call reconstruct( &
          g, r(cc, :), h, box, c_k, d_k, phi, dipole_flag, &
          energy_cbt, df_cbt, fb, fc, fd, dipoles(ind, 3:4, :), trans_matrix(ind, 4:6, :, :))

        energy = energy + energy_cbt
        df_dih = df_dih + df_cbt

        ! Angle forces
        force(bb, :) = force(bb, :) - fb
        force(cc, :) = force(cc, :) - fc
        force(dd, :) = force(dd, :) - fd
      end if
    end if

    ! Dihedral forces
    sc = v * f_dot_g / (v_sq * g_norm) - w * h_dot_g / (w_sq * g_norm)

    fa = -df_dih * g_norm * v / v_sq
    fd =  df_dih * g_norm * w / w_sq

    fb =  df_dih * sc - fa
    fc = -df_dih * sc - fd
    
    ! Subtract negative gradient
    force(aa, :) = force(aa, :) - fa
    force(bb, :) = force(bb, :) - fb
    force(cc, :) = force(cc, :) - fc
    force(dd, :) = force(dd, :) - fd
  end do
end subroutine cdf
