      subroutine RECONSTRUCT_CB
C     Subroutine for calculating forces due to protein backbone
      implicit none
      include 'system.inc'
      include 'control.inc'
      include 'iochan.h'
      include 'lists.inc'
      include 'config.inc'
      include 'model.inc'
      include 'forces.inc'
      
      integer i, ia, ib, ic, id
      real*8 vb2x,vb2y,vb2z,vb3x,vb3y
      real*8 vb3z,vb2xm,vb2ym,vb2zm
      real*8 b2mag2, b2mag, b3mag2, b3mag
      real*8 calpha, invsalpha, alpha, delta
      real*8 v1(3),v2(3), vn(3), vm(3), phi, gamma, p,g,h
      real*8  rboxi(3), rbox(3), phi_coef(5), gamma_coef(5)
      phi_coef = (/2.132464, 0.032184, -0.933094, -2.494226, -2.505892/)
      gamma_coef = (/0.711547, -0.789978, 4.300896, 7.850295, 1.938433/)
      rbox  = (/xbox, ybox, zbox/)
      rboxi = (/xboxi, yboxi, zboxi/)
C     Loop for reconstructing dipoles with single charges
      do 100 i=1, ntorl
         if(dihtypb(i).ne.5) then
            goto 100
         endif 
         phi_coef = (/rb1(i), rb2(i), rb3(i), rb4(i),rb5(i)/)
         gamma_coef = (/rbb1(i), rbb2(i), rbb3(i), rbb4(i),rbb5(i)/)

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
         

         phi   = POW_COS(calpha, phi_coef)
         gamma = POW_COS(calpha, gamma_coef)
         p = cos(phi)
         g = sin(phi)*cos(gamma)
         h = sin(phi)*sin(gamma)
!         p = 
!         g = POW_COS(calpha, n_coef)
!         h = POW_COS(calpha, m_coef)
!         p  = -sqrt(1 - g**2 - h**2 )
         
         x(id) = x(ib) + rb0(i)*(p*v2(1) + g*vn(1) + h*vm(1)) 
         y(id) = y(ib) + rb0(i)*(p*v2(2) + g*vn(2) + h*vm(2))
         z(id) = z(ib) + rb0(i)*(p*v2(3) + g*vn(3) + h*vm(3)) 
                  
         x(id)  = x(id) - xbox*anint(xboxi * x(id))
         y(id)  = y(id) - ybox*anint(yboxi * y(id))
         z(id)  = z(id) - zbox*anint(zboxi * z(id))
         vx(id) = 0d0
         vy(id) = 0d0
         vz(id) = 0d0

 100  CONTINUE
      call PBC
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
      
      FUNCTION POW_COS(a,b)

      IMPLICIT NONE
      REAL*8 POW_COS
      REAL*8, INTENT(IN):: a, b(5)
      
      POW_COS = b(1) + a*(b(2)+a*(b(3)+a*(b(4)+a*b(5))))
      END FUNCTION POW_COS
      end
      
