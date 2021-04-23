      subroutine f_intra_electro
C     Subroutine for calculating forces due to protein backbone
      use nlstuff
      implicit none
      include 'system.inc'
      include 'control.inc'

      include 'lists.inc'
      include 'config.inc'
      include 'model.inc'
      include 'forces.inc'
      
      integer i, ia, ib, ic, id, ii, jj, kk
      integer j, jb, jbeg, jend, NNN, index
     
      real*8 vb2x,vb2y,vb2z,vb3x,vb3y
      real*8 vb3z,vb2xm,vb2ym,vb2zm
      real*8 b2mag2, b2mag, b3mag2, b3mag
      real*8 calpha, invsalpha, alpha, delta
      real*8 cpsi, spsi
      real*8 theta, ctheta, stheta, dtheta
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
      real*8 r0(3), d(3), rboxi(3), rbox(3), f2(3), vr_center(3)
     $     ,r_center
      parameter(delta=0.3)
c$$$      cpsi = cos(1.390927)     
c$$$      spsi = sin(1.390927)
      cpsi = 0.177              !cos(1.77)
      spsi = sqrt(1-cpsi**2)    !sin(1.77)
c$$$          cpsi = cos(c_dip(1,1))          
c$$$          spsi = sin(c_dip(1,1))

      rcuts = 0.2d0             ! 0.1 har vært standard
C     rcuts = 0.15d0
c$$$      c_dip(2,:)=0d0
c$$$      c_dip(2,1)=0.0
c$$$      parameter(delta=0.2273d0)
c$$$      force_fac=0.11d0*qfact2/dielec

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
C     Loop for reconstructing dipoles with single charges
      
      do 100  i=1, nangl 
         if(angltypb(i).ne.2) goto 100

         ib = liang(i)
         ic = ljang(i)
         id = lkang(i)
         
         vb2x = x(ic) - x(ib)
         vb2y = y(ic) - y(ib)
         vb2z = z(ic) - z(ib)
         
         vb3x = x(id) - x(ic)
         vb3y = y(id) - y(ic)
         vb3z = z(id) - z(ic)
         
         vb2x = vb2x - xbox*anint(xboxi * vb2x)
         vb3x = vb3x - xbox*anint(xboxi * vb3x)         
         
         vb2y = vb2y - ybox*anint(yboxi * vb2y)
         vb3y = vb3y - ybox*anint(yboxi * vb3y)
         
         vb2z = vb2z - zbox*anint(zboxi * vb2z)
         vb3z = vb3z - zbox*anint(zboxi * vb3z)
         
         vb2xm = -vb2x
         vb2ym = -vb2y
         vb2zm = -vb2z
         


C     Finding angle
         b2mag2 = vb2x*vb2x + vb2y*vb2y + vb2z*vb2z
         b2mag  = sqrt(b2mag2)
         b3mag2 = vb3x*vb3x + vb3y*vb3y + vb3z*vb3z
         b3mag  = sqrt(b3mag2)

         calpha = vb2xm*vb3x + vb2ym*vb3y + vb2zm*vb3z
         calpha = calpha/(b2mag*b3mag)

         if (calpha >  1.0d0) calpha =  1.0d0
         if (calpha < -1.0d0) calpha = -1.0d0

         alpha = acos(calpha)
         invsalpha = 1.0d0/sin(alpha)
c$$$         
         fac = exp((alpha-1.73d0)/0.025d0)
         theta = -1.607d0*alpha + 0.094d0 +
     $        1.883d0/(1.0d0 + fac)
         dtheta = -1.607d0 - 1.883d0/0.025d0*fac
     $        /((1.0d0 + fac)**2)
c$$$ 
c$$$         fac=tanh(6.91*(alpha-1.9175))
c$$$         theta=0.617+1.62*fac
c$$$         dtheta= 1.62*6.91*(1-fac**2)
c$$$         fac = tanh(11.60583484*(alpha - 1.89723012))
c$$$         theta = 0.58996143 + 1.60460649*fac
c$$$         dtheta =  1.60460649*11.60583484*(1-fac**2)
c$$$ 
          
c$$$         theta=COS10(c_dip(2,:),calpha)
c$$$         dtheta=-1.0d0/invsalpha*DCOS10(c_dip(2,:),calpha)

c$$$  fac = tanh(6.9*(alpha - 1.89723012))
c$$$         theta = 0.58996143 + 1.60460649*fac
c$$$         dtheta =  1.60460649*6.9*(1-fac**2)
          
         ctheta = cos(theta)
         stheta = sin(theta)
         
         v1 = (/vb2xm, vb2ym, vb2zm/)/b2mag 
         v2 = (/vb3x, vb3y, vb3z/)/b3mag
         
         vn = invsalpha*CROSS(v1, v2)
         vm = CROSS(vn, v2)
         

C     Bending forces computation
         a11 = -calpha / b2mag2
         a12 = 1.00d0 / (b2mag*b3mag)
         a22 = -calpha / b3mag2
 

         fi(1)   = a11*vb2xm + a12*vb3x
         fi(2)   = a11*vb2ym + a12*vb3y
         fi(3)   = a11*vb2zm + a12*vb3z
         
         fiii(1) = a22*vb3x + a12*vb2xm
         fiii(2) = a22*vb3y + a12*vb2ym
         fiii(3) = a22*vb3z + a12*vb2zm         
         
         fii     = -(fi + fiii)
         
C     Construction of dipole
         r0 = (/x(ic), y(ic), z(ic)/) + 0.5d0 * (/vb3x, vb3y, vb3z/) 
         d  = 0.50d0*delta*(cpsi*v2 + spsi*(ctheta*vn + stheta*vm))
         
         r_pos(:,i) = r0 + d
         r_neg(:,i) = r0 - d
         
C     Taking care of periodicity
         r_pos(:,i) = r_pos(:,i) - rbox * anint (rboxi * r_pos(:,i))
         r_neg(:,i) = r_neg(:,i) - rbox * anint (rboxi * r_neg(:,i))
         
C     Preparation for transformation matricies D
         
         V2II(1,1) = -1.0d0 + v2(1)*v2(1)
         V2II(1,2) =          v2(1)*v2(2)
         V2II(1,3) =          v2(1)*v2(3)
         V2II(2,2) = -1.0d0 + v2(2)*v2(2)
         V2II(2,3) =          v2(2)*v2(3)
         V2II(3,3) = -1.0d0 + v2(3)*v2(3)
         V2II(2,1) =  V2II(1,2)
         V2II(3,1) =  V2II(1,3)
         V2II(3,2) =  V2II(2,3)
         
         V2II  = V2II/b3mag
         V2III = -V2II

         
         V1II(1,1) = -1.0d0 + v1(1)*v1(1)
         V1II(1,2) =          v1(1)*v1(2)
         V1II(1,3) =          v1(1)*v1(3)
         V1II(2,2) = -1.0d0 + v1(2)*v1(2)
         V1II(2,3) =          v1(2)*v1(3)
         V1II(3,3) = -1.0d0 + v1(3)*v1(3)
         V1II(2,1) =  V1II(1,2)
         V1II(3,1) =  V1II(1,3)
         V1II(3,2) =  V1II(2,3)
         
         V1II = V1II/b2mag
         V1I  = -V1II
         

         fac  = calpha*invsalpha
         
         NI   = invsalpha*(CROSS_MATRIX(V1I, v2) +
     $        fac*OUTER_PRODUCT(fi, vn)) 
         NII  = invsalpha*(CROSS_MATRIX(V1II, v2) + 
     $        CROSS_MATRIX(V2II,-v1) + fac*OUTER_PRODUCT(fii,vn))
         NIII = invsalpha*(CROSS_MATRIX(V2III, -v1) +
     $        fac*OUTER_PRODUCT(fiii, vn)) 
         
         MI   = CROSS_MATRIX(NI, v2)
         MII  = CROSS_MATRIX(NII, v2)  + CROSS_MATRIX(V2II, -vn)
         MIII = CROSS_MATRIX(NIII, v2) + CROSS_MATRIX(V2III, -vn)
         
         fac = stheta*invsalpha*dtheta
         FNI   = fac*OUTER_PRODUCT(fi, vn)
         FNII  = fac*OUTER_PRODUCT(fii, vn)
         FNIII = fac*OUTER_PRODUCT(fiii, vn)
         
         fac   = ctheta*invsalpha*dtheta
         FMI   = fac*OUTER_PRODUCT(fi, vm)
         FMII  = fac*OUTER_PRODUCT(fii, vm)
         FMIII = fac*OUTER_PRODUCT(fiii, vm)

C     Final transfer matricies
         DI(:,:,i)   = 0.50d0*delta*(             spsi*(ctheta*NI   + 
     $        stheta*MI   + FNI   - FMI))
         DII(:,:,i)  = 0.50d0*delta*(cpsi*V2II  + spsi*(ctheta*NII  + 
     $        stheta*MII  + FNII  - FMII))
         DIII(:,:,i) = 0.50d0*delta*(cpsi*V2III + spsi*(ctheta*NIII + 
     $        stheta*MIII + FNIII - FMIII))
         
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

C         write(*,*) "i=", i, "j=", j
C     do 110 i=1,ntorl-3-1
C     Here we need to implement neighbourlist
C     if(dihtypb(i).ne.4) goto 110
         
C     do 120 j = i+3, ntorl
C     if(dihtypb(j).ne.4) goto 120

C     Simple neighbourlist approach
C     Extra
c$$$         vr_center = 0.5*(r_neg(:,j)+r_pos(:,j)-r_neg(:,i)-r_pos(:,i))
c$$$         vr_center = vr_center - rbox*anint(rboxi*vr_center)
c$$$         r_center = NORM2(vr_center)
c$$$         f2 = 0.5*12d0*force_fac*((0.5d0/r_center)**12)*
c$$$     $        vr_center*r_center**(-2)
c$$$         eb = eb + force_fac*((0.5d0/r_center)**12)
C     

         
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

         
c$$$C     Hallo
c$$$                     rnn = 1.0d0/NORM2(vrnn)
c$$$C         rnp = 1.0d0/NORM2(vrnp)
c$$$C         rpn = 1.0d0/NORM2(vrpn)
c$$$         rpp = 1.0d0/NORM2(vrpp)
c$$$
c$$$         ef_nn =  force_fac*rnn
c$$$C         ef_np = -force_fac*rnp
c$$$C         ef_pn = -force_fac*rpn
c$$$         ef_pp =  force_fac*rpp
c$$$         
c$$$         ff_nn =  ef_nn*rnn**2
c$$$C         ff_np =  ef_np*rnp**2
c$$$C         ff_pn =  ef_pn*rpn**2
c$$$         ff_pp =  ef_pp*rpp**2
c$$$
c$$$C         rnn = NORM2(vrnn)
c$$$         rnp = NORM2(vrnp)
c$$$         rpn = NORM2(vrpn)
c$$$C         rpp = NORM2(vrpp)
c$$$
c$$$!     if(rnn.le.rcuts) write(*,*) i,j,'rnn=', rnn
c$$$          !  if(rpn.le.rcuts) write(*,*) i,j,'rpn=', rpn
c$$$          !  if(rnp.le.rcuts) write(*,*) i,j,'rnp=', rnp
c$$$          !  if(rpp.le.rcuts) write(*,*) i,j,'rpp=', rpp
c$$$      
c$$$          
c$$$c$$$            ef_nn =  force_fac*COLN(rnn, rcuts)
c$$$            ef_np = -force_fac*COLN(rnp, rcuts)
c$$$            ef_pn = -force_fac*COLN(rpn, rcuts)
c$$$c$$$            ef_pp =  force_fac*COLN(rpp, rcuts)
c$$$         
c$$$
c$$$c$$$
c$$$c$$$            ff_nn = -force_fac*DCOLN(rnn, rcuts)/rnn
c$$$            ff_np =  force_fac*DCOLN(rnp, rcuts)/rnp
c$$$            ff_pn =  force_fac*DCOLN(rpn, rcuts)/rpn
c$$$C           ff_pp = -force_fac*DCOLN(rpp, rcuts)/rpp



C     Added
c$$$         f_neg(:,i)= f_neg(:,i)-f2
c$$$         f_neg(:,j)= f_neg(:,j)+f2
c$$$         f_pos(:,i)= f_pos(:,i)-f2
c$$$         f_pos(:,j)= f_pos(:,j)+f2

         
C     Neg-Neg
            f = vrnn*ff_nn 
            
            f_neg(:, i) = f_neg(:, i) - f
            f_neg(:, j) = f_neg(:, j) + f
            
C     Neg-Pos
            f = vrnp*ff_np 
            
            f_neg(:, i) = f_neg(:, i) - f
            f_pos(:, j) = f_pos(:, j) + f
            
C     Pos-Neg
            f = vrpn*ff_pn
            
            f_pos(:, i) = f_pos(:, i) - f
            f_neg(:, j) = f_neg(:, j) + f
            
C     Pos-Pos
            f = vrpp*ff_pp 
            
            f_pos(:, i) = f_pos(:, i) - f
            f_pos(:, j) = f_pos(:, j) + f

            eb = eb + ef_nn + ef_np + ef_pn + ef_pp
            
C120        CONTINUE
 110      CONTINUE
      
C     Mapping forces
      do 130  i=1, nangl 
         if(angltypb(i).ne.2) goto 130
         ib = liang(i)
         ic = ljang(i)
         id = lkang(i)
         
C     Total force
         f = 0.5d0*(f_pos(:,i) + f_neg(:,i))

         fxatom(ic) = fxatom(ic) + f(1)
         fxatom(id) = fxatom(id) + f(1)
         
         fyatom(ic) = fyatom(ic) + f(2)
         fyatom(id) = fyatom(id) + f(2)
        
         fzatom(ic) = fzatom(ic) + f(3)
         fzatom(id) = fzatom(id) + f(3)
         
C     Force difference

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

      
C     Mapping torque
      
      deallocate(r_neg)
      deallocate(r_pos)
      deallocate(f_neg)
      deallocate(f_pos)
      deallocate(DI)
      deallocate(DII)
      deallocate(DIII)


      return
      CONTAINS
      FUNCTION CROSS(a,b)
C     Crossproduct between two vectors
      IMPLICIT NONE
      REAL*8 CROSS(3)
      REAL*8, INTENT(IN):: a(3), b(3)
      
      CROSS(1) = a(2)*b(3) - a(3)*b(2)
      CROSS(2) = a(3)*b(1) - a(1)*b(3)
      CROSS(3) = a(1)*b(2) - a(2)*b(1)
      END FUNCTION CROSS
      
      FUNCTION CROSS_MATRIX(A,b)
C     Note: A is matrix while b is vector
      IMPLICIT NONE
      REAL*8 CROSS_MATRIX(3,3)
      REAL*8, INTENT(IN):: A(:,:), b(:)
      CROSS_MATRIX(:,1) = A(:,2)*b(3) - A(:,3)*b(2)
      CROSS_MATRIX(:,2) = A(:,3)*b(1) - A(:,1)*b(3)
      CROSS_MATRIX(:,3) = A(:,1)*b(2) - A(:,2)*b(1)
      END FUNCTION CROSS_MATRIX

 
      FUNCTION OUTER_PRODUCT(a,b)  
      
      IMPLICIT NONE
      REAL*8 OUTER_PRODUCT(3,3)
      REAL*8, INTENT(IN):: a(3), b(3)
      
      OUTER_PRODUCT(1,:) = a(1)*b
      OUTER_PRODUCT(2,:) = a(2)*b
      OUTER_PRODUCT(3,:) = a(3)*b
      END FUNCTION OUTER_PRODUCT

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
     


C      if(DABS(r-a)<1E-6) then
C         COLN = n/(r*(1.0+n))
C      else
         COLN = (r**10-a**10)/(r**(11)-a**(11))
C      endif
    
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


      
      
c$$$      FUNCTION DCOLN(r,a,n)
c$$$
c$$$      implicit none
c$$$      REAL*8 DCOLN
c$$$      REAL*8  r, a
c$$$      INTEGER n
c$$$      INTENT(IN) :: r, a, n
c$$$
c$$$      if(DABS(r-a)<1E-4) then
c$$$         DCOLN = -n**2/(r**2 *(1.0+n))
c$$$      else 
c$$$      DCOLN = (n*r**(n-1)*(r**(n+1)-a**(n+1))-
c$$$     $        (r**n-a**n)*(n+1)*r**n)/(r**(n+1)-a**(n+1))**2
c$$$      endif
c$$$    
c$$$      END FUNCTION DCOLN

      end
      
