      subroutine fdihedral_vdv
C     Subroutine for calculating forces due to protein backbone

      implicit none
      include 'system.inc'
      include 'control.inc'
      include 'iochan.h'
      include 'lists.inc'
      include 'config.inc'
      include 'model.inc'
      include 'forces.inc'
      
      integer i, ia, ib, ic, id, ii,jj,m
      real*8 vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y
      real*8 vb3z,vb2xm,vb2ym,vb2zm,r6
      
      real*8 ax,ay,az,bx,by,bz,rasq,rbsq,rgsq,rg
      real*8 rginv,ra2inv,rb2inv,rabinv,c,s,theta
      real*8 fg,hg,fga,hgb,gaa,gbb,dtfx,dtfy,dtfz,dtgx,dtgy,dtgz
      real*8 dthx,dthy,dthz,sx2,sy2,sz2,p
      real*8 rboxi(3), rbox(3),vrij(3),fij(3), rij, C_ij(5)
      real*8 p_, ddf1_, df1_,df,dvdr      
      real*8 SMALL, SMALLER, phi0, TOLERANCE
      parameter (SMALL=0.001d0, SMALLER=0.00001d0, phi0=0d0)
      parameter (TOLERANCE=0.05d0)

      rbox  = (/xbox, ybox, zbox/)
      rboxi = (/xboxi, yboxi, zboxi/)
      do 100 i=1, ntorl
         if(dihtypb(i).ne.6) goto 100
         p=0d0

         ia = litor(i)
         ib = ljtor(i)
         ic = lktor(i)
         id = lltor(i)
         

         vb1x =  -(x(ib) - x(ia))
         vb1y =  -(y(ib) - y(ia))
         vb1z =  -(z(ib) - z(ia))

         vb2x =  x(ic) - x(ib)
         vb2y =  y(ic) - y(ib)
         vb2z =  z(ic) - z(ib)
         
         vb3x =  x(id) - x(ic)
         vb3y =  y(id) - y(ic)
         vb3z =  z(id) - z(ic)
         
         
         vb2xm = -vb2x
         vb2ym = -vb2y
         vb2zm = -vb2z
         
         
         vb1x = vb1x - xbox * anint (xboxi * vb1x)
         vb2x = vb2x - xbox * anint (xboxi * vb2x)
         vb3x = vb3x - xbox * anint (xboxi * vb3x)
         
         vb1y = vb1y - ybox * anint (yboxi * vb1y)
         vb2y = vb2y - ybox * anint (yboxi * vb2y)
         vb3y = vb3y - ybox * anint (yboxi * vb3y)
         
         vb1z = vb1z - zbox * anint (zboxi * vb1z)
         vb2z = vb2z - zbox * anint (zboxi * vb2z)
         vb3z = vb3z - zbox * anint (zboxi * vb3z)

         vb2xm = -vb2x
         vb2ym = -vb2y
         vb2zm = -vb2z
         
         vrij = (/-vb1x+vb2x+vb3x,-vb1y+vb2y+vb3y,-vb1z+vb2z+vb3z/)
         rij  = NORM2(vrij)

         r6 = (rb0(i)/rij)**6

         fij  = vrij/(rij**2)*24*rb1(i)*r6*(2*r6-1)
         p    = 4*rb1(i)*r6*(r6-1)

         fxatom(ia) = fxatom(ia) - fij(1)
         fyatom(ia) = fyatom(ia) - fij(2)
         fzatom(ia) = fzatom(ia) - fij(3)
         
         fxatom(ib) = fxatom(ib) 
         fyatom(ib) = fyatom(ib) 
         fzatom(ib) = fzatom(ib) 
         
         fxatom(ic) = fxatom(ic) 
         fyatom(ic) = fyatom(ic) 
         fzatom(ic) = fzatom(ic) 
         
         fxatom(id) = fxatom(id) + fij(1)
         fyatom(id) = fyatom(id) + fij(2)
         fzatom(id) = fzatom(id) + fij(3)
         
         vtors = vtors + p

 100  CONTINUE
      
      return
      CONTAINS
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
