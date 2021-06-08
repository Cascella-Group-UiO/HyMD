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

  function cross_matrix(M, v) result(matrix)
    ! The i-th row of the (o)utput matrix is the cross product 
    ! between the i-th row of the input (M)atrix and the input (v)ector.
    real(8), dimension(3,3), intent(in) :: M
    real(8), dimension(3),   intent(in) :: v
    real(8), dimension(3,3)             :: o

    o(:,1) = M(:,2) * v(3) - M(:,3) * v(2)
    o(:,2) = M(:,3) * v(1) - M(:,1) * v(3)
    o(:,3) = M(:,1) * v(2) - M(:,2) * v(1)
  end function

  function outer_product(vector1, vector2) result(matrix)
    ! The (o)utput is a matrix where the i-th row is the multiplication
    ! of the i-th component of vector1 and the full vector2
    real(8), dimension(3), intent(in) :: vector1, vector2
    real(8), dimension(3,3)           :: o

    o(1,:) = vector1(1) * vector2
    o(2,:) = vector1(2) * vector2
    o(3,:) = vector1(3) * vector2
  end function outer_product

subroutine reconstruct(dipole_positions, f, g, g_norm)
    ! Called inside dihedral routine?
    implicit none

    real(8), dimension(3), intent(in out) :: dipole_positions
    real(8), dimension(3), intent(in) :: f, g
    real(8), dimension(3, 3), intent(out) :: D_a, D_b, D_c
    real(8), parameter, intent(in) :: g_norm

    real(8), parameter :: delta = 0.3, phi = 1.390927, cos_phi = cos(1.390927), sin_phi = sin(1.390927)

    ! cos_phi = 0.177
    ! sin_phi = sqrt(1-cos_phi**2)

    ! We import f and g from the dihedral subroutine
    g = -g
    ! Finding angle
    f_norm = norm2(f)
    g_norm = norm2(g)
    cos_gamma = dot_product(f, g) / (f_norm * g_norm)

    if (cos_gamma >  1.d0) then
        cos_gamma =  1.d0
    if (cos_gamma < -1.d0) then 
        cos_gamma = -1.d0

    gamma_ang = acos(cos_gamma)
    sin_gamma = sin(gamma_ang)

    ! This function needs to be fit again    
    fac = exp((gamma_ang - 1.73d0) / 0.025d0)
    theta = -1.607d0*gamma_ang + 0.094d0 + 1.883d0/(1.0d0 + fac)
    d_theta = -1.607d0 - 1.883d0/0.025d0 * fac / ((1.0d0 + fac)**2)
   
    cos_theta = cos(theta)
    sin_theta = sin(theta)

    ! f_versor == f_hat
    f_hat = f / f_norm
    ! g_versor == v_hat
    v_hat = g / g_norm
    n = cross(f_hat, v_hat) / sin_gamma
    m = cross(n, v_hat)
  
    ! Bending forces == f_gamma_i in the paper
    force_on_a = (v_hat / g_norm - cos_gamma * f) / (f_norm * f_norm)
    force_on_c = (f_hat / f_norm - cos_gamma * g) / (g_norm * g_norm)
    force_on_b = -(force_on_a + force_on_c)

    ! Dipole construction, import r2
    r0 = r2 + 0.5d0 * g
    d  = 0.5d0 * delta * (cos_phi * v + sin_phi * cos_theta * n + sin_phi * sin_theta * m)

    r_pos = r0 + d
    r_neg = r0 - d

    ! Taking care of periodicity
    r_pos = r_pos - box * nint(r_pos / box)
    r_neg = r_neg - box * nint(r_neg / box)

    ! Preparation of transformation matrix D
    do i = 1, 3
        do j = 1, 3
            V_b(i, j) = v_hat(i) * v_hat(j)
            F_b(i, j) = f_hat(i) * f_hat(j)
            if (i == j) then
                V_b(i, j) = V_b(i, j) - 1.d0
                F_b(i, j) = F_b(i, j) - 1.d0
            V_b(i, j) = V_b(i, j) / g_norm
            F_b(i, j) = F_b(i, j) / f_norm

    V_c = -V_b
    F_a = -F_b
    
    ! V1 = F
    ! V2 = V
    fac  = cos_gamma / sin_gamma

    ! N_a = inv_sin_gamma*(cross_matrix(V1I, v2) + fac*outer_product(fi, vn)) 
    ! N_b = inv_sin_gamma*(cross_matrix(V1II, v2) + cross_matrix(V2II,-v1) + fac*outer_product(fii,vn))
    ! N_c = inv_sin_gamma*(cross_matrix(V2III, -v1) + fac*outer_product(fiii, vn)) 

    ! Don't know why v1 (f_hat) is multiplied with -1?
    ! Oh ok, because of cross product order
    N_a = (fac * outer_product(force_on_a, n) + cross_matrix(F_a, v_hat)) / sin_gamma
    N_b = (fac * outer_product(force_on_b, n) + cross_matrix(F_b, v_hat) - cross_matrix(V_b, f_hat) + ) / sin_gamma
    N_c = (fac * outer_product(force_on_c, n) - cross_matrix(V_c, f_hat)) / sin_gamma

    M_a = cross_matrix(N_a, v_hat)
    M_b = cross_matrix(N_b, v_hat) - cross_matrix(V_b, n)
    M_c = cross_matrix(N_c, v_hat) - cross_matrix(V_c, n)
 
    ! Check the following are correct
    fac = sin_theta * dtheta / sin_gamma
    FN_a = fac * outer_product(force_on_a, n)
    FN_b = fac * outer_product(force_on_b, n)
    FN_c = fac * outer_product(force_on_c, n)

    fac = cos_theta * dtheta / sin_gamma
    FM_a = fac * outer_product(force_on_a, m)
    FM_b = fac * outer_product(force_on_b, m)
    FM_c = fac * outer_product(force_on_c, m)

    ! Final transfer matricies
    D_a = 0.5d0 * delta * (sin_phi * (cos_theta * NI + sin_theta * MI   + FNI   - FMI))
    D_b = 0.5d0 * delta * (cos_phi * V2II  + sin_phi * (cos_theta * NII + sin_theta * MII  + FNII  - FMII))
    D_c = 0.5d0 * delta * (cos_phi * V2III + sin_phi * (cos_theta * NIII + sin_theta * MIII + FNIII - FMIII))

    100  CONTINUE


    i=1
    INDEX=2
    do 110
        if(listdip(INDEX).eq.-1) EXIT

        if(listdip(INDEX).eq.0)then
         i     = i + 1
         INDEX = INDEX + 1
         goto 110
        endif
        j     = listdip(INDEX)
        INDEX = INDEX + 1

        vrnn = r_neg(:, j) - r_neg(:, i)
        vrnp = r_pos(:, j) - r_neg(:, i)
        vrpn = r_neg(:, j) - r_pos(:, i)
        vrpp = r_pos(:, j) - r_pos(:, i)
         
        vrnn = vrnn - rbox*anint(rboxi*vrnn)
        vrnp = vrnp - rbox*anint(rboxi*vrnp)
        vrpn = vrpn - rbox*anint(rboxi*vrpn)
        vrpp = vrpp - rbox*anint(rboxi*vrpp)

        rnn = NORM2(vrnn)
        rnp = NORM2(vrnp)
        rpn = NORM2(vrpn)
        rpp = NORM2(vrpp)

        ef_nn =  force_fac/rnn
        ef_np = -force_fac*COLN(rnp, rcuts)
        ef_pn = -force_fac*COLN(rpn, rcuts)
        ef_pp =  force_fac/rpp

        ff_nn =  ef_nn/rnn**2
        ff_np =  force_fac*DCOLN(rnp, rcuts)/rnp
        ff_pn =  force_fac*DCOLN(rpn, rcuts)/rpn
        ff_pp =  ef_pp/rpp**2

        ! Neg-Neg
        f = vrnn*ff_nn 
        
        f_neg(:, i) = f_neg(:, i) - f
        f_neg(:, j) = f_neg(:, j) + f
        
        ! Neg-Pos
        f = vrnp*ff_np 
        
        f_neg(:, i) = f_neg(:, i) - f
        f_pos(:, j) = f_pos(:, j) + f
        
        ! Pos-Neg
        f = vrpn*ff_pn
        
        f_pos(:, i) = f_pos(:, i) - f
        f_neg(:, j) = f_neg(:, j) + f
        
        ! Pos-Pos
        f = vrpp*ff_pp 
        
        f_pos(:, i) = f_pos(:, i) - f
        f_pos(:, j) = f_pos(:, j) + f

        eb = eb + ef_nn + ef_np + ef_pn + ef_pp
        
    110      CONTINUE
   
! Mapping forces
    do 130  i=1, nangl 
        if(angltypb(i).ne.2) goto 130
        ib = liang(i)
        ic = ljang(i)
        id = lkang(i)

        ! Total force
        f = 0.5d0*(f_pos(:,i) + f_neg(:,i))

        fxatom(ic) = fxatom(ic) + f(1)
        fxatom(id) = fxatom(id) + f(1)

        fyatom(ic) = fyatom(ic) + f(2)
        fyatom(id) = fyatom(id) + f(2)

        fzatom(ic) = fzatom(ic) + f(3)
        fzatom(id) = fzatom(id) + f(3)

        ! Force difference

        f = f_pos(:,i) - f_neg(:,i)

        fi   = MATMUL(DI(:,:,i)  ,f)
        fii  = MATMUL(DII(:,:,i) ,f)
        fiii = MATMUL(DIII(:,:,i),f)

        fxatom(ib) = fxatom(ib) + fi(1)
        fyatom(ib) = fyatom(ib) + fi(2)
        fzatom(ib) = fzatom(ib) + fi(3)

        fxatom(ic) = fxatom(ic) + fii(1)
        fyatom(ic) = fyatom(ic) + fii(2)
        fzatom(ic) = fzatom(ic) + fii(3)

        fxatom(id) = fxatom(id) + fiii(1)
        fyatom(id) = fyatom(id) + fiii(2)
        fzatom(id) = fzatom(id) + fiii(3)
   130     CONTINUE

   
! Mapping torque
   
    deallocate(r_neg)
    deallocate(r_pos)
    deallocate(f_neg)
    deallocate(f_pos)
    deallocate(DI)
    deallocate(DII)
    deallocate(DIII)
    return

    contains

    function col_pot(r, rcut, n)  

    implicit none
    real*8 col_pot
    real*8, intent(in):: r, rcut
    integer n

    col_pot = (r**n - rcut**n)/(r**(n+1) - rcut**(n+1))
    end function col_pot

    function d_col_pot(r, rcut, n)  

    implicit none
    real*8 d_col_pot
    real*8, intent(in):: r, rcut
    integer n

    d_col_pot = (n*r**(n-1)*(r**(n+1)-rcut**(n+1))
    $     -(r**n-rcut**n)*(n+1)*r**n)
    $     /((r**(n+1)-rcut**(n+1))**2)
    end function d_col_pot

    function col(r, rcut)  

    implicit none
    real*8 col
    real*8, intent(in):: r, rcut

    col = 1d0/(r+rcut)
    end function col

    function dcol(r, rcut)  

    implicit none
    real*8 dcol
    real*8, intent(in):: r, rcut

    dcol = -1d0/((r+rcut)**2)
    end function dcol

    function col2(r, rcuts)  

    implicit none
    real*8 col2
    real*8, intent(in):: r, rcuts

    col2 = (rcuts+r)/(rcuts**2+rcuts*r+r**2)
    end function col2 

    function dcol2(r, rcuts)  

    implicit none
    real*8 dcol2
    real*8, intent(in):: r, rcuts

    dcol2 = -r*(2*rcuts+r)/((rcuts**2+rcuts*r+r**2)**2)
    end function dcol2


    function coln(r,a)

    implicit none
    real*8 coln
    real*8, intent(in):: r, a



    !  if(dabs(r-a)<1e-6) then
    !     coln = n/(r*(1.0+n))
    !  else
      coln = (r**10-a**10)/(r**(11)-a**(11))
    !  endif

    end function coln 

    function dcoln(r,a)

    implicit none
    real*8 dcoln
    real*8  r, a, x
    integer n
    intent(in) :: r, a


    x=r/a
    dcoln = -(1d0/a**2) *( x**9 *(x**9 + 2 *x**8 + 3 *x**7 + 4 *x**6+5
    $     *x**5 + 6 *x**4 + 7 *x**3 + 8 *x**2 + 9*x + 10))/
    $     (x**10 + x**9 + x**8 + x**7 + x**6 + x**5 +
    $     x**4 + x**3 + x**2 + x + 1)**2


    end function dcoln

    function cos10(a,c)  
        
    implicit none
    real*8 cos10
    real*8, intent(in):: a(10), c

    cos10 = a(1)+c*(a(2)+c*(a(3)+
    $     c*(a(4)+c*(a(5)+c*(a(6)+c*(a(7)+
    $     c*(a(8)+c*(a(9)+c*a(10)))))))))
    end function cos10
    function dcos10(a,c)  
        
    implicit none
    real*8 dcos10
    real*8, intent(in):: a(10), c

    dcos10 = a(2)+c*(2*a(3)+
    $     c*(3*a(4)+c*(4*a(5)+c*(5*a(6)+c*(6*a(7)+
    $     c*(7*a(8)+c*(8*a(9)+9*c*a(10))))))))
    end function dcos10

end subroutine backbone_dipole_reconstruction
