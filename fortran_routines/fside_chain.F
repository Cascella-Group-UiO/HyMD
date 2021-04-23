      subroutine fside_chain
C     Subroutine for calculating forces due to protein backbone
      use nlstuff
      implicit none
      include 'system.inc'
      include 'control.inc'
      include 'iochan.h'
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
      
      real*8 a, a11, a22, a12, rmin, p, g, h, dp, dg, dh
      real*8 f(3), V1I(3,3), V1II(3,3), V2II(3,3), V2III(3,3)
      real*8 MI(3,3), MII(3,3), MIII(3,3), NI(3,3), NII(3,3), NIII(3,3)
      real*8 FVNMI(3,3), FVNMII(3,3), FVNMIII(3,3)
      real*8 DI(3,3), DII(3,3), DIII(3,3)
      real*8 v1(3),v2(3), vn(3), vm(3), fac, fi(3), fii(3), fiii(3)
      real*8 rboxi(3), rbox(3)
      real*8 phi_coef(5), gamma_coef(5), phi, gamma, dphi, dgamma
      real*8 sphi, sgamma, cphi, cgamma

      rbox  = (/xbox, ybox, zbox/)
      rboxi = (/xboxi, yboxi, zboxi/)
     
      f     = 0.0d0
      DI    = 0.0d0
      DII   = 0.0d0
      DIII  = 0.0d0
      
      
      rbox  = (/xbox, ybox, zbox/)
      rboxi = (/xboxi, yboxi, zboxi/)
C     Loop for reconstructing dipoles with single charges
      do 100 i=1, ntorl
         if(dihtypb(i).ne.5) then
            goto 100
         endif 
         phi_coef = (/rb1(i), rb2(i), rb3(i), rb4(i),rb5(i)/)
         gamma_coef = (/rbb1(i),rbb2(i),rbb3(i),rbb4(i),rbb5(i)/)
C         write(*,*) gamma_coef
         
         ia = litor(i)
         ib = ljtor(i)
         ic = lltor(i)
         id = lktor(i)     

         
         vb2x = x(ib) - x(ia)
         vb2y = y(ib) - y(ia)
         vb2z = z(ib) - z(ia)
         
         vb3x = x(ic) - x(ib)
         vb3y = y(ic) - y(ib)
         vb3z = z(ic) - z(ib)
         
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
         
         v1 = (/vb2xm, vb2ym, vb2zm/)/b2mag 
         v2 = (/vb3x, vb3y, vb3z/)/b3mag
         
         vn = invsalpha*CROSS(v1, v2)
         vm = CROSS(vn, v2)


C     Projection contributions to v, n and m
C         p  =   POW_COS(calpha, v_coef)
C         dp = D_POW_COS(calpha, v_coef)
         
         phi    = POW_COS(calpha, phi_coef)
         dphi   = D_POW_COS(calpha, phi_coef)
         
         gamma  = POW_COS(calpha, gamma_coef)
         dgamma = D_POW_COS(calpha, gamma_coef)

         sphi   = sin(phi)
         sgamma = sin(gamma)
         
         cphi   = cos(phi)
         cgamma = cos(gamma)
         

         p  = cphi
         dp = -dphi*sphi
         
         g  = sphi*cgamma
         dg = dphi*cphi*cgamma - dgamma*sphi*sgamma
         
         h  = sphi*sgamma
         dh = dphi*cphi*sgamma + dgamma*sphi*cgamma 
          
C         g  =   POW_COS(calpha, n_coef)
C         dg = D_POW_COS(calpha, n_coef)

C         h  =   POW_COS(calpha, m_coef)
C         dh = D_POW_COS(calpha, m_coef)

C         p  = -sqrt(1-g**2-h**2 )
C         dp = -(g*dg + h*dh)/p

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
         
         MI   = CROSS_MATRIX(NI,   v2)
         MII  = CROSS_MATRIX(NII,  v2) + CROSS_MATRIX(V2II,  -vn)
         MIII = CROSS_MATRIX(NIII, v2) + CROSS_MATRIX(V2III, -vn)
         
         FVNMI   = OUTER_PRODUCT(fi,   dp*v2 + dg*vn + dh*vm)
         FVNMII  = OUTER_PRODUCT(fii,  dp*v2 + dg*vn + dh*vm)
         FVNMIII = OUTER_PRODUCT(fiii, dp*v2 + dg*vn + dh*vm)
      


C     Final transfer matricies
         DI       = rb0(i)*(          g*NI   + h*MI   + FVNMI)
         DII      = rb0(i)*(p*V2II +  g*NII  + h*MII  + FVNMII)
         DIII     = rb0(i)*(p*V2III + g*NIII + h*MIII + FVNMIII)
         DII(1,1) = DII(1,1) + 1d0
         DII(2,2) = DII(2,2) + 1d0
         DII(3,3) = DII(3,3) + 1d0
         
        


C     Distribution of forces 
         f = (/ fxatom(id), fyatom(id), fzatom(id) /)
C         write(*,*) f
         fi   = MATMUL(DI,   f)
         fii  = MATMUL(DII,  f)
         fiii = MATMUL(DIII, f)

         fxatom(ia) = fxatom(ia) + fi(1)
         fyatom(ia) = fyatom(ia) + fi(2)
         fzatom(ia) = fzatom(ia) + fi(3)
         
         fxatom(ib) = fxatom(ib) + fii(1)
         fyatom(ib) = fyatom(ib) + fii(2)
         fzatom(ib) = fzatom(ib) + fii(3)

         fxatom(ic) = fxatom(ic) + fiii(1)
         fyatom(ic) = fyatom(ic) + fiii(2)
         fzatom(ic) = fzatom(ic) + fiii(3)

         fxatom(id) = 0d0
         fyatom(id) = 0d0
         fzatom(id) = 0d0

 100  CONTINUE

      
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
      REAL*8, INTENT(IN):: A(3,3), b(3)
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


      FUNCTION POW_COS(a,b)

      IMPLICIT NONE
      REAL*8 POW_COS
      REAL*8, INTENT(IN):: a, b(5)
      
      POW_COS = b(1) + a*(b(2)+a*(b(3)+a*(b(4)+a*b(5))))
      END FUNCTION POW_COS
      
      FUNCTION D_POW_COS(a,b)

      IMPLICIT NONE
      REAL*8 D_POW_COS
      REAL*8, INTENT(IN):: a, b(5)
      
      D_POW_COS = b(2)+a*(2*b(3)+a*(3*b(4)+4*a*b(5)))
      END FUNCTION D_POW_COS
      end
      
