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
  ! The i-th column of the output matrix is the cross product 
  ! between the i-th column of the input matrix and the input vector.
  real(8), dimension(3,3), intent(in) :: matrix
  real(8), dimension(3),   intent(in) :: vector
  real(8), dimension(3,3)             :: output

  output(:,1) = matrix(:,2) * vector(3) - matrix(:,3) * vector(2)
  output(:,2) = matrix(:,3) * vector(1) - matrix(:,1) * vector(3)
  output(:,3) = matrix(:,1) * vector(2) - matrix(:,2) * vector(1)
end function

function outer_product(vector1, vector2) result(output)
  ! The i-th row of the output matrix is vector2 
  ! multiplied by the i-th component of vector1.
  real(8), dimension(3), intent(in) :: vector1, vector2
  real(8), dimension(3,3)           :: output

  output(1,:) = vector1(1) * vector2
  output(2,:) = vector1(2) * vector2
  output(3,:) = vector1(3) * vector2
end function outer_product

! REfactoring in FORTRAN is kinda bad :(
! subroutine angle_force()
! end subroutine angle_force

subroutine reconstruct(rab, rb, rcb, box, phi, energy, df_cbt, fa, fc, dipole, trans_matrix)
  real(8), dimension(3), intent(in)  :: rab, rb, rcb, box
  real(8),               intent(in)  :: phi, gamm, dipole_flag
  real(8),               intent(out) :: energy, df_cbt
  real(8), dimension(3), intent(out) :: fa, fc
  real(4), dimension(2, 3), intent(in out) :: dipole
  real(4), dimension(3, 3, 3), intent(in out) :: trans_matrix

  integer :: i, j
  real(8) :: k, gamma_0. dk, dg
  real(8) :: theta, d_theta, cos_theta, sin_theta, fac
  real(8) :: cos_gamma, sin_gamma, cos2
  real(8), dimension(3) :: w, v, n, m, r0, d, fb 
  real(8), dimension(3, 3) :: W_a, W_b, V_b, V_c
  real(8), dimension(3, 3) :: N_a, N_b, N_c
  real(8), dimension(3, 3) :: M_a, M_b, M_c
  real(8), dimension(3, 3) :: FN_a, fN_b, fN_c
  real(8), dimension(3, 3) :: FM_a, FM_b, FM_c
  real(8), parameter :: delta = 0.3d0, cos_phi = cos(1.390927), sin_phi = sin(1.390927) 
  ! cos_phi = 0,17890101
  ! sin_phi = 0,983867079

  ! 1 - Angle forces calculation
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
    gamma_0 = gamma_0 + c_g(i + 1) * (1.d0 + cos(i * phi + d_g(i + 1)))

    dk = dk + i * c_k(i + 1) * sin(i * phi + d_k(i + 1))
    dg = dg + i * c_g(i + 1) * sin(i * phi + d_g(i + 1))
  end do

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
    sin_gamma = sin(theta)

    ! Bending forces == f_gamma_i in the paper
    ! 1/sin(γ) ∂cos(γ)/∂γ
    fa = (v - cos_gamma * w) / (norm_a * sin_gamma)
    fc = (w - cos_gamma * v) / (norm_c * sin_gamma)
    fb = -(fa + fc)

    ! CBT energy and force factors
    df_ang = -k * (gamm - gamma_0) 
    var_sq = (gamm - gamma_0)**2
    energy = 0.5d0 * k * var_sq      
    df_cbt = 0.5d0 * dk * var_sq + df_ang * dg
    
    ! Exit subroutine if we only need the forces
    if (dipole_flag == 0) then
      fa = -df_ang * fa
      fc = -df_ang * fc
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
    d  = 0.5d0 * delta * (cos_phi * v + sin_phi * (cos_theta * n + sin_theta * m))

    dipole(1, :) = r0 + d
    dipole(2, :) = r0 - d
    
    ! PBC
    dipole(1, :) = dipole(1, :) - box * nint(dipole(1, :) / box)
    dipole(2, :) = dipole(2, :) - box * nint(dipole(2, :) / box)

    ! Set up transfer matrices
    do i = 1, 3
        do j = 1, 3
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
    ! fac = cos_gamma
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

    ! fac = cos_theta * d_theta
    FM_a = cos_theta * d_theta * outer_product(fa, m)
    FM_b = cos_theta * d_theta * outer_product(fb, m)
    FM_c = cos_theta * d_theta * outer_product(fc, m)

    ! Final transfer matrices D_i
    ! 0.5 cause we have two equally distant points
    trans_matrix(1, :, :) = 0.5d0 * delta * (                sin_phi * (cos_theta * N_a + sin_theta * M_a + FN_a - FM_a))
    trans_matrix(2, :, :) = 0.5d0 * delta * (cos_phi * V_b + sin_phi * (cos_theta * N_b + sin_theta * M_b + FN_b - FM_b))
    trans_matrix(3, :, :) = 0.5d0 * delta * (cos_phi * V_c + sin_phi * (cos_theta * N_c + sin_theta * M_c + FN_c - FM_c))

    ! Final angle forces
    fa = -df_ang * fa
    fc = -df_ang * fc
    end if
end subroutine reconstruct
end module dipole_reconstruction
