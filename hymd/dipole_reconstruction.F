module dipole_reconstruction
  implicit none
  contains

  function cross(vector1, vector2) result(vector3)
    real(8), dimension(3), intent(in) :: vector1, vector2
    real(8), dimension(3) :: vector3 
    
    vector3(1) = vector1(2) * vector2(3) - vector1(3) * vector2(2)
    vector3(2) = vector1(3) * vector2(1) - vector1(1) * vector2(3)
    vector3(3) = vector1(1) * vector2(2) - vector1(2) * vector2(1)
  end function

  function cross_matrix(M, v) result(matrix)
    ! M is a matrix, v is a vector
    real(8), dimension(3,3), intent(in) :: M
    real(8), dimension(3),   intent(in) :: b
    real(8), dimension(3,3)             :: matrix

    matrix(:,1) = M(:,2) * v(3) - M(:,3) * v(2)
    matrix(:,2) = M(:,3) * v(1) - M(:,1) * v(3)
    matrix(:,3) = M(:,1) * v(2) - M(:,2) * v(1)
  end function

  function outer_product(vector1, vector2) result(matrix)
    real(8), dimension(3), intent(in) :: vector1, vector2
    real(8), dimension(3,3)           :: matrix

    matrix(1,:) = a(1)*b
    matrix(2,:) = a(2)*b
    matrix(3,:) = a(3)*b
  end function outer_product

subroutine backbone_dipole_reconstruction
    ! Called inside dihedral
    implicit none

    integer i, ia, ib, ic, id, ii, jj, kk
    integer j, jb, jbeg, jend, NNN, index

    real*8 vb2x,vb2y,vb2z,vb3x,vb3y
    real*8 vb3z,vb2xm,vb2ym,vb2zm
    real*8 b2mag2, b2mag, b3mag2, b3mag
    real*8 cos_gamma, inv_sin_gamma, angle, delta
    real*8 cos_phi, sin_phi
    real*8 theta, cos_theta, sin_theta, dtheta
    real*8 rpp, rnp, rpn, rnn
    real*8 vrnn(3), vrnp(3),vrpn(3),vrpp(3)
    real*8 a, a11, a22, a12, rmin
    real*8 force_fac, p, fac
    real*8 ff_nn,ff_np,ff_pn,ff_pp
    real*8 ef_nn,ef_np,ef_pn,ef_pp, rcuts
    real*8 f(3), V1I(3,3), V1II(3,3), V2II(3,3), V2III(3,3)
    real*8 MI(3,3), MII(3,3), MIII(3,3), NI(3,3), NII(3,3), NIII(3,3)
    real*8 FNI(3,3),FNII(3,3),FNIII(3,3),FMI(3,3),FMII(3,3),FMIII(3,3)
    real*8, ALLOCATABLE :: r_neg(:,:), r_pos(:,:)
    real*8, ALLOCATABLE :: f_neg(:,:), f_pos(:,:)
    real*8, ALLOCATABLE :: DI(:,:,:), DII(:,:,:), DIII(:,:,:)
    real*8 v1(3),v2(3), vn(3), vm(3), fi(3), fii(3), fiii(3)
    real*8 r0(3), d(3), rboxi(3), rbox(3), f2(3), vr_center(3), r_center

    real(8), parameter :: delta = 0.3, phi = 1.390927

    cos_phi = cos(1.390927)     
    sin_phi = sin(1.390927)

    ! cos_phi = 0.177               !cos(1.77)
    ! sin_phi = sqrt(1-cos_phi**2)  !sin(1.77)

    !       cos_phi = cos(c_dip(1,1))          
    !       sin_phi = sin(c_dip(1,1))

    ! What's this?
    rcuts = 0.2d0             ! 0.1 har vært standard
    ! rcuts = 0.15d0
    ! c_dip(2,:)=0d0
    ! c_dip(2,1)=0.0
    ! parameter(delta=0.2273d0)
    ! force_fac=0.11d0*qfact2/dielec
    ! What's this?
    force_fac=(0.020819434*3.6/delta)**2*qfact2/dielec

    allocate(r_neg(3,nangl)) 
    allocate(r_pos(3,nangl))
    allocate(f_neg(3,nangl))
    allocate(f_pos(3,nangl))
    allocate(DI(3,3,nangl))
    allocate(DII(3,3,nangl))
    allocate(DIII(3,3,nangl))
    r_neg = 0.0d0
    r_pos = 0.0d0
    f_neg = 0.0d0
    f_pos = 0.0d0
    f     = 0.0d0
    DI    = 0.0d0
    DII   = 0.0d0
    DIII  = 0.0d0
    p     = 0.0d0
    fac   = 0.0d0
   

   
    rbox  = (/xbox, ybox, zbox/)
    rboxi = (/xboxi, yboxi, zboxi/)
    ! Loop for reconstructing dipoles with single charges
   
    do 100  i=1, nangl 
        if(angltypb(i).ne.2) goto 100

        ib = liang(i)
        ic = ljang(i)
        id = lkang(i)

        ! -f
        f = [r(ic:,) - r(ib:,)]
        vb2x = x(ic) - x(ib)
        vb2y = y(ic) - y(ib)
        vb2z = z(ic) - z(ib)
        ! -g
        g = [r(id:,) - r(ic:,)]
        vb3x = x(id) - x(ic)
        vb3y = y(id) - y(ic)
        vb3z = z(id) - z(ic)

        vb2x = vb2x - xbox*anint(xboxi * vb2x)
        vb3x = vb3x - xbox*anint(xboxi * vb3x)         

        vb2y = vb2y - ybox*anint(yboxi * vb2y)
        vb3y = vb3y - ybox*anint(yboxi * vb3y)

        vb2z = vb2z - zbox*anint(zboxi * vb2z)
        vb3z = vb3z - zbox*anint(zboxi * vb3z)

        do i = 1, 3
        f(i) = f(i) - box(i) * nint(f(i) / box(i))
        g(i) = g(i) - box(i) * nint(g(i) / box(i))
        h(i) = h(i) - box(i) * nint(h(i) / box(i))
        end do

        ! f
        vb2xm = -vb2x
        vb2ym = -vb2y
        vb2zm = -vb2z
      


        ! Finding angle
        ! Norms
        b2mag2 = vb2x*vb2x + vb2y*vb2y + vb2z*vb2z
        b2mag  = sqrt(b2mag2)
        b3mag2 = vb3x*vb3x + vb3y*vb3y + vb3z*vb3z
        b3mag  = sqrt(b3mag2)

        cos_gamma = dot_product(f, g)
        cos_gamma = vb2xm*vb3x + vb2ym*vb3y + vb2zm*vb3z
        cos_gamma = cos_gamma/(b2mag*b3mag)

        if (cos_gamma >  1.0d0) cos_gamma =  1.0d0
        if (cos_gamma < -1.0d0) cos_gamma = -1.0d0

        angle = acos(cos_gamma)
        inv_sin_gamma = 1.0d0 / sin(angle)
        !      
        fac = exp((gamma-1.73d0)/0.025d0)
        theta = -1.607d0*angle + 0.094d0 + 1.883d0/(1.0d0 + fac)
        dtheta = -1.607d0 - 1.883d0/0.025d0 * fac / ((1.0d0 + fac)**2)
!
!      fac=tanh(6.91*(gamma-1.9175))
!      theta=0.617+1.62*fac
!      dtheta= 1.62*6.91*(1-fac**2)
!      fac = tanh(11.60583484*(gamma - 1.89723012))
!      theta = 0.58996143 + 1.60460649*fac
!      dtheta =  1.60460649*11.60583484*(1-fac**2)
!
       
!      theta=COS10(c_dip(2,:),cos_gamma)
!      dtheta=-1.0d0/inv_sin_gamma*DCOS10(c_dip(2,:),cos_gamma)

!ac = tanh(6.9*(gamma - 1.89723012))
!      theta = 0.58996143 + 1.60460649*fac
!      dtheta =  1.60460649*6.9*(1-fac**2)

        cos_theta = cos(theta)
        sin_theta = sin(theta)

        v1 = (/vb2xm, vb2ym, vb2zm/)/b2mag 
        v2 = (/vb3x, vb3y, vb3z/)/b3mag

        ! vn == n versor
        vn = inv_sin_gamma*CROSS(v1, v2)
        vm = CROSS(vn, v2)
      

        ! Bending forces computation
        ! == f_gamma_i in the paper
        a11 = -cos_gamma / b2mag2
        a12 = 1.00d0 / (b2mag*b3mag)
        a22 = -cos_gamma / b3mag2

        fi(1)   = a11*vb2xm + a12*vb3x
        fi(2)   = a11*vb2ym + a12*vb3y
        fi(3)   = a11*vb2zm + a12*vb3z

        fiii(1) = a22*vb3x + a12*vb2xm
        fiii(2) = a22*vb3y + a12*vb2ym
        fiii(3) = a22*vb3z + a12*vb2zm         

        fii     = -(fi + fiii)

        ! Construction of dipole
        r0 = (/x(ic), y(ic), z(ic)/) + 0.5d0 * (/vb3x, vb3y, vb3z/) 
        d  = 0.50d0*delta*(cos_phi*v2 + sin_phi*(cos_theta*vn + sin_theta*vm))

        r_pos(:,i) = r0 + d
        r_neg(:,i) = r0 - d

        ! Taking care of periodicity
        r_pos(:,i) = r_pos(:,i) - rbox * anint (rboxi * r_pos(:,i))
        r_neg(:,i) = r_neg(:,i) - rbox * anint (rboxi * r_neg(:,i))

        ! Preparation of transformation matrix D

        V2II(1,1) = -1.0d0 + v2(1)*v2(1)
        V2II(1,2) =          v2(1)*v2(2)
        V2II(1,3) =          v2(1)*v2(3)
        V2II(2,2) = -1.0d0 + v2(2)*v2(2)
        V2II(2,3) =          v2(2)*v2(3)
        V2II(3,3) = -1.0d0 + v2(3)*v2(3)
        ! Symmetric matrix : see eq. S15
        V2II(2,1) = V2II(1,2)
        V2II(3,1) = V2II(1,3)
        V2II(3,2) = V2II(2,3)

        V2II  = V2II/b3mag
        V2III = -V2II


        V1II(1,1) = -1.0d0 + v1(1)*v1(1)
        V1II(1,2) =          v1(1)*v1(2)
        V1II(1,3) =          v1(1)*v1(3)
        V1II(2,2) = -1.0d0 + v1(2)*v1(2)
        V1II(2,3) =          v1(2)*v1(3)
        V1II(3,3) = -1.0d0 + v1(3)*v1(3)
        ! Symmetric matrix : see eq. S15
        V1II(2,1) =  V1II(1,2)
        V1II(3,1) =  V1II(1,3)
        V1II(3,2) =  V1II(2,3)

        V1II = V1II/b2mag
        V1I  = -V1II


        fac  = cos_gamma*inv_sin_gamma

        NI   = inv_sin_gamma*(CROSS_MATRIX(V1I, v2) + fac*OUTER_PRODUCT(fi, vn)) 
        NII  = inv_sin_gamma*(CROSS_MATRIX(V1II, v2) + CROSS_MATRIX(V2II,-v1) + fac*OUTER_PRODUCT(fii,vn))
        NIII = inv_sin_gamma*(CROSS_MATRIX(V2III, -v1) + fac*OUTER_PRODUCT(fiii, vn)) 

        MI   = CROSS_MATRIX(NI, v2)
        MII  = CROSS_MATRIX(NII, v2)  + CROSS_MATRIX(V2II, -vn)
        MIII = CROSS_MATRIX(NIII, v2) + CROSS_MATRIX(V2III, -vn)

        fac = sin_theta*inv_sin_gamma*dtheta
        FNI   = fac*OUTER_PRODUCT(fi, vn)
        FNII  = fac*OUTER_PRODUCT(fii, vn)
        FNIII = fac*OUTER_PRODUCT(fiii, vn)

        fac   = cos_theta*inv_sin_gamma*dtheta
        FMI   = fac*OUTER_PRODUCT(fi, vm)
        FMII  = fac*OUTER_PRODUCT(fii, vm)
        FMIII = fac*OUTER_PRODUCT(fiii, vm)

        ! Final transfer matricies
        DI(:,:,i)   = 0.50d0*delta*(sin_phi*(cos_theta*NI + sin_theta*MI   + FNI   - FMI))
        DII(:,:,i)  = 0.50d0*delta*(cos_phi*V2II  + sin_phi*(cos_theta*NII + sin_theta*MII  + FNII  - FMII))
        DIII(:,:,i) = 0.50d0*delta*(cos_phi*V2III + sin_phi*(cos_theta*NIII + sin_theta*MIII + FNIII - FMIII))

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

!     write(*,*) "i=", i, "j=", j
! do 110 i=1,ntorl-3-1
! Here we need to implement neighbourlist
! if(dihtypb(i).ne.4) goto 110
      
! do 120 j = i+3, ntorl
! if(dihtypb(j).ne.4) goto 120

! Simple neighbourlist approach
! Extra
!      vr_center = 0.5*(r_neg(:,j)+r_pos(:,j)-r_neg(:,i)-r_pos(:,i))
!      vr_center = vr_center - rbox*anint(rboxi*vr_center)
!      r_center = NORM2(vr_center)
!      f2 = 0.5*12d0*force_fac*((0.5d0/r_center)**12)*
!  $        vr_center*r_center**(-2)
!      eb = eb + force_fac*((0.5d0/r_center)**12)
! 

      
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


        !  Hallo
        !                  rnn = 1.0d0/NORM2(vrnn)
        !      rnp = 1.0d0/NORM2(vrnp)
        !      rpn = 1.0d0/NORM2(vrpn)
        !      rpp = 1.0d0/NORM2(vrpp)
        !
        !      ef_nn =  force_fac*rnn
        !      ef_np = -force_fac*rnp
        !      ef_pn = -force_fac*rpn
        !      ef_pp =  force_fac*rpp
        !      
        !      ff_nn =  ef_nn*rnn**2
        !      ff_np =  ef_np*rnp**2
        !      ff_pn =  ef_pn*rpn**2
        !      ff_pp =  ef_pp*rpp**2
        !
        !      rnn = NORM2(vrnn)
        !      rnp = NORM2(vrnp)
        !      rpn = NORM2(vrpn)
        !      rpp = NORM2(vrpp)
        !
        !   if(rnn.le.rcuts) write(*,*) i,j,'rnn=', rnn
        !       !  if(rpn.le.rcuts) write(*,*) i,j,'rpn=', rpn
        !       !  if(rnp.le.rcuts) write(*,*) i,j,'rnp=', rnp
        !       !  if(rpp.le.rcuts) write(*,*) i,j,'rpp=', rpp
        !   
        !       
        !          ef_nn =  force_fac*COLN(rnn, rcuts)
        !         ef_np = -force_fac*COLN(rnp, rcuts)
        !         ef_pn = -force_fac*COLN(rpn, rcuts)
        !          ef_pp =  force_fac*COLN(rpp, rcuts)
        !      
        !
        !
        !          ff_nn = -force_fac*DCOLN(rnn, rcuts)/rnn
        !         ff_np =  force_fac*DCOLN(rnp, rcuts)/rnp
        !         ff_pn =  force_fac*DCOLN(rpn, rcuts)/rpn
        !        ff_pp = -force_fac*DCOLN(rpp, rcuts)/rpp



        ! Added
        !      f_neg(:,i)= f_neg(:,i)-f2
        !      f_neg(:,j)= f_neg(:,j)+f2
        !      f_pos(:,i)= f_pos(:,i)-f2
        !      f_pos(:,j)= f_pos(:,j)+f2


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

    CONTAINS

    FUNCTION COL_POT(r, rcut, n)  

    IMPLICIT NONE
    REAL*8 COL_POT
    REAL*8, INTENT(IN):: r, rcut
    INTEGER n

    COL_POT = (r**n - rcut**n)/(r**(n+1) - rcut**(n+1))
    END FUNCTION COL_POT

    FUNCTION D_COL_POT(r, rcut, n)  

    IMPLICIT NONE
    REAL*8 D_COL_POT
    REAL*8, INTENT(IN):: r, rcut
    INTEGER n

    D_COL_POT = (n*r**(n-1)*(r**(n+1)-rcut**(n+1))
    $     -(r**n-rcut**n)*(n+1)*r**n)
    $     /((r**(n+1)-rcut**(n+1))**2)
    END FUNCTION D_COL_POT

    FUNCTION COL(r, rcut)  

    IMPLICIT NONE
    REAL*8 COL
    REAL*8, INTENT(IN):: r, rcut

    COL = 1d0/(r+rcut)
    END FUNCTION COL

    FUNCTION DCOL(r, rcut)  

    IMPLICIT NONE
    REAL*8 DCOL
    REAL*8, INTENT(IN):: r, rcut

    DCOL = -1d0/((r+rcut)**2)
    END FUNCTION DCOL

    FUNCTION COL2(r, rcuts)  

    IMPLICIT NONE
    REAL*8 COL2
    REAL*8, INTENT(IN):: r, rcuts

    COL2 = (rcuts+r)/(rcuts**2+rcuts*r+r**2)
    END FUNCTION COL2 

    FUNCTION DCOL2(r, rcuts)  

    IMPLICIT NONE
    REAL*8 DCOL2
    REAL*8, INTENT(IN):: r, rcuts

    DCOL2 = -r*(2*rcuts+r)/((rcuts**2+rcuts*r+r**2)**2)
    END FUNCTION DCOL2


    FUNCTION COLN(r,a)

    IMPLICIT NONE
    REAL*8 COLN
    REAL*8, INTENT(IN):: r, a



    !  if(DABS(r-a)<1E-6) then
    !     COLN = n/(r*(1.0+n))
    !  else
      COLN = (r**10-a**10)/(r**(11)-a**(11))
    !  endif

    END FUNCTION COLN 

    FUNCTION DCOLN(r,a)

    implicit none
    REAL*8 DCOLN
    REAL*8  r, a, x
    INTEGER n
    INTENT(IN) :: r, a


    x=r/a
    DCOLN = -(1d0/a**2) *( x**9 *(x**9 + 2 *x**8 + 3 *x**7 + 4 *x**6+5
    $     *x**5 + 6 *x**4 + 7 *x**3 + 8 *x**2 + 9*x + 10))/
    $     (x**10 + x**9 + x**8 + x**7 + x**6 + x**5 +
    $     x**4 + x**3 + x**2 + x + 1)**2


    END FUNCTION DCOLN

    FUNCTION COS10(a,c)  
        
    IMPLICIT NONE
    REAL*8 COS10
    REAL*8, INTENT(IN):: a(10), c

    COS10 = a(1)+c*(a(2)+c*(a(3)+
    $     c*(a(4)+c*(a(5)+c*(a(6)+c*(a(7)+
    $     c*(a(8)+c*(a(9)+c*a(10)))))))))
    END FUNCTION COS10
    FUNCTION DCOS10(a,c)  
        
    IMPLICIT NONE
    REAL*8 DCOS10
    REAL*8, INTENT(IN):: a(10), c

    DCOS10 = a(2)+c*(2*a(3)+
    $     c*(3*a(4)+c*(4*a(5)+c*(5*a(6)+c*(6*a(7)+
    $     c*(7*a(8)+c*(8*a(9)+9*c*a(10))))))))
    END FUNCTION DCOS10




!   FUNCTION DCOLN(r,a,n)
!
!   implicit none
!   REAL*8 DCOLN
!   REAL*8  r, a
!   INTEGER n
!   INTENT(IN) :: r, a, n
!
!   if(DABS(r-a)<1E-4) then
!      DCOLN = -n**2/(r**2 *(1.0+n))
!   else 
!   DCOLN = (n*r**(n-1)*(r**(n+1)-a**(n+1))-
!  $        (r**n-a**n)*(n+1)*r**n)/(r**(n+1)-a**(n+1))**2
!   endif
! 
!   END FUNCTION DCOLN
end subroutine backbone_dipole_reconstruction
