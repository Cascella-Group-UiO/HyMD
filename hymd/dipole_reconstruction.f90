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

function cross_matrix(M, v) result(output)
  ! The i-th row of the output matrix is the cross product 
  ! between the i-th row of the input (M)atrix and the input (v)ector.
  real(8), dimension(3,3), intent(in) :: M
  real(8), dimension(3),   intent(in) :: v
  real(8), dimension(3,3)             :: output

  output(:,1) = M(:,2) * v(3) - M(:,3) * v(2)
  output(:,2) = M(:,3) * v(1) - M(:,1) * v(3)
  output(:,3) = M(:,1) * v(2) - M(:,2) * v(1)
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

subroutine reconstruct(g, h, g_norm, r_b, box, dipole, trans_matrix)
  real(8), intent(in) :: g_norm
  real(4), dimension(3), intent(in) :: r_b
  real(8), dimension(3), intent(in) :: g, h, box
  real(8), dimension(2, 3), intent(out) :: dipole
  real(8), dimension(3, 3, 3), intent(out) :: trans_matrix

  integer :: i, j
  real(8) :: gamma_ang, cos_gamma, sin_gamma, theta, d_theta, cos_theta, sin_theta, fac, h_norm
  real(8), dimension(3) :: w, v, n, m, r0, d, force_on_a, force_on_b, force_on_c 
  real(8), dimension(3, 3) :: W_a, W_b, V_b, V_c, N_a, N_b, N_c, M_a, M_b, M_c, FN_a, FN_b, FN_c, FM_a, FM_b, FM_c
  real(8), parameter :: delta = 0.3, cos_phi = cos(1.390927), sin_phi = sin(1.390927) 

  ! cos_phi = 0,17890101
  ! sin_phi = 0,983867079

  h_norm = norm2(h)
  cos_gamma = dot_product(g, h) / (g_norm * h_norm)

  ! Needed?
  if (cos_gamma >  1.d0) then
      cos_gamma =  1.d0
  end if
  if (cos_gamma < -1.d0) then 
      cos_gamma = -1.d0
  end if

  gamma_ang = acos(cos_gamma)
  sin_gamma = sin(gamma_ang)

  ! This function needs to be fit again    
  fac = exp((gamma_ang - 1.73d0) / 0.025d0)
  theta = -1.607d0 * gamma_ang + 0.094d0 + 1.883d0 / (1.0d0 + fac)
  d_theta = -1.607d0 - 1.883d0 / 0.025d0 * fac / ((1.0d0 + fac)**2)
  
  cos_theta = cos(theta)
  sin_theta = sin(theta)

  ! g_versor == w 
  w = g / g_norm
  ! h_versor == v
  v = h / h_norm
  n = cross(w, v) / sin_gamma
  m = cross(n, v)
  
  ! Bending forces == f_gamma_i in the paper
  force_on_a = (v / h_norm - cos_gamma * g) / (g_norm * g_norm)
  force_on_c = (w / g_norm - cos_gamma * h) / (h_norm * h_norm)
  force_on_b = -(force_on_a + force_on_c)

  ! Dipole coordinates
  r0 = r_b + 0.5d0 * h
  d  = 0.5d0 * delta * (cos_phi * v + sin_phi * (cos_theta * n + sin_theta * m))

  dipole(1, :) = r0 + d
  dipole(2, :) = r0 - d
  
  ! PBC
  dipole(1, :) = dipole(1, :) - box * nint(dipole(1, :) / box)
  dipole(2, :) = dipole(2, :) - box * nint(dipole(2, :) / box)

  do i = 1, 3
      do j = 1, 3
          V_b(i, j) = v(i) * v(j)
          W_b(i, j) = w(i) * w(j)
          if (i == j) then
              V_b(i, j) = V_b(i, j) - 1.d0
              W_b(i, j) = W_b(i, j) - 1.d0
          end if
          V_b(i, j) = V_b(i, j) / h_norm
          W_b(i, j) = W_b(i, j) / g_norm
      end do 
  end do

  V_c = -V_b
  W_a = -W_b
  
  ! Last term is 0 for N_a, second term is 0 for N_c (S19)
  fac  = cos_gamma / sin_gamma
  N_a = (fac * outer_product(force_on_a, n) + cross_matrix(W_a, v)                       ) / sin_gamma
  N_b = (fac * outer_product(force_on_b, n) + cross_matrix(W_b, v) - cross_matrix(V_b, w)) / sin_gamma
  N_c = (fac * outer_product(force_on_c, n)                        - cross_matrix(V_c, w)) / sin_gamma

  M_a = cross_matrix(N_a, v)
  M_b = cross_matrix(N_b, v) - cross_matrix(V_b, n)
  M_c = cross_matrix(N_c, v) - cross_matrix(V_c, n)
 
  ! A lot of terms in S10 go away because, 
  ! since phi = const., ∂phi/∂gamma = 0
  fac = sin_theta * d_theta / sin_gamma
  FN_a = fac * outer_product(force_on_a, n)
  FN_b = fac * outer_product(force_on_b, n)
  FN_c = fac * outer_product(force_on_c, n)

  fac = cos_theta * d_theta / sin_gamma
  FM_a = fac * outer_product(force_on_a, m)
  FM_b = fac * outer_product(force_on_b, m)
  FM_c = fac * outer_product(force_on_c, m)

  ! Final transfer matrices D_i
  trans_matrix(1, :, :) = 0.5d0 * delta * (                sin_phi * (cos_theta * N_a + sin_theta * M_a + FN_a - FM_a))
  trans_matrix(2, :, :) = 0.5d0 * delta * (cos_phi * V_b + sin_phi * (cos_theta * N_b + sin_theta * M_b + FN_b - FM_b))
  trans_matrix(3, :, :) = 0.5d0 * delta * (cos_phi * V_c + sin_phi * (cos_theta * N_c + sin_theta * M_c + FN_c - FM_c))

end subroutine reconstruct
end module dipole_reconstruction