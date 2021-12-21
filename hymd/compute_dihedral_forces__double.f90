subroutine cdf_d(force, r, dipoles, transfer_matrix, box, a, b, c, d, coeff, dtype, bb_index, dipole_flag, energy)
  use dipole_reconstruction_d
  implicit none

  real(8), intent(in out) :: force(:,:)
  real(8), intent(in) :: r(:,:)
  real(8), intent(in out) :: dipoles(:,:,:)
  real(8), intent(in out) :: transfer_matrix(:,:,:,:)
  real(8), intent(in) :: box(:)
  integer, intent(in) :: a(:), b(:), c(:), d(:), dtype(:), bb_index(:), dipole_flag
  real(4), intent(in) :: coeff(:,:,:)
  real(8), intent(out) :: energy

  integer :: ind, aa, bb, cc, dd, i
  integer, dimension(2) :: c_shape
  real(8), dimension(3) :: f, g, h, v, w, sc, fa, fb, fc, fd
  real(8), dimension(5) :: c_v, c_k, c_coil, d_coil, d_v, d_k
  ! real(8), dimension(5) :: c_g, d_g
  real(8) :: energy_cbt, df_cbt
  real(8) :: force_const, eq_value
  real(8) :: g_norm, v_sq, w_sq, f_dot_g, h_dot_g
  real(8) :: df_dih, df_ang, cos_phi, sin_phi, phi

  energy = 0.d0
  force = 0.d0
  dipoles = 0.d0
  transfer_matrix = 0.d0

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
    sin_phi = dot_product(w, f) * g_norm
    phi = atan2(sin_phi, cos_phi)

    f_dot_g = dot_product(f, g)
    h_dot_g = dot_product(h, g)

    ! Cosine series, V_prop
    if (dtype(ind) == 0 .or. dtype(ind) == 1) then
      df_dih = 0.d0
      c_v = coeff(ind, 1, :)
      d_v = coeff(ind, 2, :)
      call cosine_series(c_v, d_v, phi, energy, df_dih)

      c_coil = coeff(ind, 3, :)
      d_coil = coeff(ind, 4, :)
      if (count(c_coil == 0) /= size(c_coil) .and. &
          count(d_coil == 0) /= size(d_coil)) then
        call cosine_series(c_coil, d_coil, phi, energy, df_dih)
      end if
    end if

    ! CBT potential
    if (dtype(ind) == 1) then
      ! V = V_prop + k * (gamma - gamma_0)**2

      c_k = coeff(ind, 5, :)
      d_k = coeff(ind, 6, :)

      ! These are needed if gamma_0 is expressed as cosine series, not implemented
      ! c_g = coeff(ind, 7, :)
      ! d_g = phase(ind, 8, :)

      call reconstruct( &
        f, r(bb, :), -g, box, c_k, d_k, phi, dipole_flag, &
        energy_cbt, df_cbt, fa, fb, fc, dipoles(ind, 1:2, :), transfer_matrix(ind, 1:3, :, :))

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
          energy_cbt, df_cbt, fb, fc, fd, dipoles(ind, 3:4, :), transfer_matrix(ind, 4:6, :, :))

        energy = energy + energy_cbt
        df_dih = df_dih + df_cbt

        ! Angle forces
        force(bb, :) = force(bb, :) - fb
        force(cc, :) = force(cc, :) - fc
        force(dd, :) = force(dd, :) - fd
      end if
    end if

    ! Improper dihedrals, needs to be fixed, I don't like it
    if (dtype(ind) == 2) then
      eq_value = coeff(ind, 1, 1)
      force_const = coeff(ind, 1, 2)
      df_dih = force_const * (phi - eq_value)
      energy = energy + 0.5 * force_const * (phi - eq_value) ** 2
    end if

    ! Dihedral forces
    sc = v * f_dot_g / (v_sq * g_norm) - w * h_dot_g / (w_sq * g_norm)

    fa = -df_dih * g_norm * v / v_sq
    fd =  df_dih * g_norm * w / w_sq
    fb =  df_dih * sc - fa
    fc = -df_dih * sc - fd

    ! Subtract negative gradient
    force(aa, :) = force(aa, :) + fa
    force(bb, :) = force(bb, :) + fb
    force(cc, :) = force(cc, :) + fc
    force(dd, :) = force(dd, :) + fd

  end do
end subroutine cdf_d
