module dipole_reconstruction
implicit none

contains
function cross(vector1, vector2) result(vector3)
  real(8), dimension(3), intent(in) :: vector1, vector2
  real(8), dimension(3)             :: vector3

  vector3(1) = vector1(2) * vector2(3) - vector1(3) * vector2(2)
  vector3(2) = vector1(3) * vector2(1) - vector1(1) * vector2(3)
  vector3(3) = vector1(1) * vector2(2) - vector1(2) * vector2(1)
end function

function cross_matrix(matrix, vector) result(output)
  ! The i-th row of the output matrix is the cross product
  ! between the i-th row of the input matrix and the input vector.
  real(8), dimension(3,3), intent(in) :: matrix
  real(8), dimension(3),   intent(in) :: vector
  real(8), dimension(3,3)             :: output

  output(1, :) = cross(matrix(1, :), vector)
  output(2, :) = cross(matrix(2, :), vector)
  output(3, :) = cross(matrix(3, :), vector)
end function

function outer_product(vector1, vector2) result(output)
  ! The i-th row of the output matrix is vector2
  ! multiplied by the i-th component of vector1.
  real(8), dimension(3), intent(in) :: vector1, vector2
  real(8), dimension(3,3)           :: output

  output(1, :) = vector1(1) * vector2
  output(2, :) = vector1(2) * vector2
  output(3, :) = vector1(3) * vector2
end function outer_product

subroutine cosine_series(c_n, d_n, phi, energy, dE_dphi)
  real(8), dimension(:), intent(in) :: c_n, d_n
  real(8), intent(in) :: phi
  real(8), intent(in out) :: energy, dE_dphi
  integer :: i

  do i = 0, size(c_n) - 1
    energy = energy + c_n(i + 1) * (1.d0 + cos(i * phi - d_n(i + 1)))
    dE_dphi = dE_dphi - i * c_n(i + 1) * sin(i * phi - d_n(i + 1))
  end do

end subroutine cosine_series

subroutine reconstruct(rab, rb, rcb, box, c_k, d_k, phi, dipole_flag, energy_cbt, df_cbt, fa, fb, fc, dipole, transfer_matrix)
  real(8), dimension(3), intent(in)  :: rab, rcb, box
  real(4), dimension(3), intent(in)  :: rb
  real(8), dimension(:), intent(in)  :: c_k, d_k
  ! real(8), dimension(:), intent(in)  :: c_g, d_g
  real(8),               intent(in)  :: phi
  integer                            :: dipole_flag
  real(8),               intent(out) :: energy_cbt, df_cbt
  real(8), dimension(3), intent(out) :: fa, fb, fc
  real(4), dimension(2, 3), intent(in out) :: dipole
  real(4), dimension(3, 3, 3), intent(in out) :: transfer_matrix

  integer :: i, j
  real(8) :: k, gamma_0, dk, dg, norm_a, norm_c, df_ang, var_sq
  real(8) :: theta, d_theta, cos_theta, sin_theta, fac
  real(8) :: gamm, cos_gamma, sin_gamma, cos2, cos_phi, sin_phi
  real(8), dimension(3) :: w, v, n, m, r0, d
  real(8), dimension(3, 3) :: W_a, W_b, V_b, V_c
  real(8), dimension(3, 3) :: N_a, N_b, N_c
  real(8), dimension(3, 3) :: M_a, M_b, M_c
  real(8), dimension(3, 3) :: FN_a, fN_b, fN_c
  real(8), dimension(3, 3) :: FM_a, FM_b, FM_c
  real(8), parameter :: delta = 0.3d0 , cos_psi = cos(1.392947), sin_psi = sin(1.392947), small = 0.001d0

  cos_phi = cos(phi)
  sin_phi = sin(phi)
  ! real(8), parameter :: small = 0.001d0
  ! cos_phi = 0,17890101
  ! sin_phi = 0,983867079
  ! energy_cbt = 0.d0

  ! 1 - Angle forces calculation
  ! Levitt-Warshel
  ! gamma_0 = 106 - 13 * cos(phi - 45)
  gamma_0 = 1.85d0 - 0.227d0 * cos(phi - 0.785d0)
  dg = 0.227d0 * sin(phi - 0.785d0)
  ! Not used in the end?
  ! We use another Fourier expansion?

  k = 0.d0
  dk = 0.d0
  ! gamma_0 = 0.d0
  ! dg = 0.d0

  call cosine_series(c_k, d_k, phi, k, dk)
  ! call cosine_series(c_g, d_g, gamma_0, dg)

  norm_a = norm2(rab)
  norm_c = norm2(rcb)

  ! w == ea, v == ec (in angle routine)
  w = rab / norm_a
  v = rcb / norm_c

  cos_gamma = dot_product(w, v)
  cos2 = cos_gamma * cos_gamma

  ! This prevents sin_gamma == 0
  if (cos2 < 1.0) then
    gamm = acos(cos_gamma)
    sin_gamma = sqrt(1 - cos2)

    if (sin_gamma < 0.1) then
      print *, "DIHEDRAL ROUTINE WARNING (bending potential):"
      print '(a, f5.2, a)', "The angle γ =", gamm, " is too close to 0 or π."
      print *, "There's probably something wrong with the simulation. Setting sin(γ) = 0.1"
      sin_gamma = 0.1
    end if

    ! Bending "forces" == f_gamma_i in the paper
    ! 1/sin(γ) ∂cos(γ)/∂γ
    fa = (v - cos_gamma * w) / norm_a
    fc = (w - cos_gamma * v) / norm_c

    fa = -fa / sin_gamma
    fc = -fc / sin_gamma

    fb = -(fa + fc)

    ! CBT energy and force factors
    df_ang = k * (gamm - gamma_0)
    var_sq = (gamm - gamma_0)**2

    energy_cbt = 0.5d0 * k * var_sq
    ! Positive gradient, add to V_prop gradient
    df_cbt = 0.5d0 * dk * var_sq - df_ang * dg

    ! Exit subroutine if we only need the forces
    if (dipole_flag == 0) then
      fa = df_ang * fa
      fb = df_ang * fb
      fc = df_ang * fc
      return
    end if

    ! 2 - Dipole reconstruction
    ! θ(γ)
    ! This function needs to be fit again
    fac = exp((gamm - 1.73d0) / 0.025d0)
    theta = -1.607d0 * gamm + 0.094d0 + 1.883d0 / (1.d0 + fac)
    d_theta = -1.607d0 - 1.883d0 / 0.025d0 * fac / ((1.d0 + fac)**2)
    cos_theta = cos(theta)
    sin_theta = sin(theta)

    n = cross(w, v) / sin_gamma
    m = cross(n, v)

    ! Dipole coordinates
    r0 = rb + 0.5d0 * rcb
    ! From Michele's paper, it's wrong in Sigbjorn's
    d  = 0.5d0 * delta * (cos_psi * v + sin_psi * (cos_theta * n + sin_theta * m))

    dipole(1, :) = r0 + d
    dipole(2, :) = r0 - d

    ! PBC
    dipole(1, :) = dipole(1, :) - box * nint(dipole(1, :) / box)
    dipole(2, :) = dipole(2, :) - box * nint(dipole(2, :) / box)

    ! Set up transfer matrices
    do j = 1, 3
      do i = 1, 3
        V_b(i, j) = v(i) * v(j)
        W_b(i, j) = w(i) * w(j)
        if (i == j) then
          V_b(i, j) = V_b(i, j) - 1.d0
          W_b(i, j) = W_b(i, j) - 1.d0
        end if
      end do
    end do

    V_b = V_b / norm_c
    W_b = W_b / norm_a

    V_c = -V_b
    W_a = -W_b

    ! Last term is 0 for N_a, second term is 0 for N_c (S19)
    ! Minus in the last term because inverse cross_matrix
    ! 1 / sin(γ) is already inside fa, fb, and fc
    N_a = (cos_gamma * outer_product(fa, n) + cross_matrix(W_a, v)                       ) / sin_gamma
    N_b = (cos_gamma * outer_product(fb, n) + cross_matrix(W_b, v) - cross_matrix(V_b, w)) / sin_gamma
    N_c = (cos_gamma * outer_product(fc, n)                        - cross_matrix(V_c, w)) / sin_gamma

    M_a = cross_matrix(N_a, v)
    M_b = cross_matrix(N_b, v) - cross_matrix(V_b, n)
    M_c = cross_matrix(N_c, v) - cross_matrix(V_c, n)

    ! A lot of terms in (S10) go away because ∂φ/∂γ = 0,
    ! since φ = const.
    ! 1 / sin(γ) is already inside fa, fb, and fc
    FN_a = sin_theta * d_theta * outer_product(fa, n)
    FN_b = sin_theta * d_theta * outer_product(fb, n)
    FN_c = sin_theta * d_theta * outer_product(fc, n)

    FM_a = cos_theta * d_theta * outer_product(fa, m)
    FM_b = cos_theta * d_theta * outer_product(fb, m)
    FM_c = cos_theta * d_theta * outer_product(fc, m)

    ! Final transfer matrices D_i
    ! 0.5 cause we have two equally distant points
    transfer_matrix(1, :, :) = 0.5d0 * delta * (                sin_psi * (cos_theta * N_a + sin_theta * M_a + FN_a - FM_a))
    transfer_matrix(2, :, :) = 0.5d0 * delta * (cos_psi * V_b + sin_psi * (cos_theta * N_b + sin_theta * M_b + FN_b - FM_b))
    transfer_matrix(3, :, :) = 0.5d0 * delta * (cos_psi * V_c + sin_psi * (cos_theta * N_c + sin_theta * M_c + FN_c - FM_c))

    ! Final angle forces
    fa = df_ang * fa
    fb = df_ang * fb
    fc = df_ang * fc
    end if
end subroutine reconstruct
end module dipole_reconstruction
