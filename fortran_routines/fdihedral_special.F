      subroutine fdihedral_special   
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
      real*8 fdih1x, fdih1y, fdih1z, fdih2x, fdih2y, fdih2z
      real*8 fdih3x, fdih3y, fdih3z, fdih4x, fdih4y, fdih4z
      real*8 vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y
      real*8 vb3z,vb2xm,vb2ym,vb2zm
      real*8 d_phi(5),K_i(5)
      real*8 ax,ay,az,bx,by,bz,rasq,rbsq,rgsq,rg
      real*8 rginv,ra2inv,rb2inv,rabinv,c,s,theta
      real*8 fg,hg,fga,hgb,gaa,gbb,dtfx,dtfy,dtfz,dtgx,dtgy,dtgz
      real*8 dthx,dthy,dthz,sx2,sy2,sz2,p
      real*8 rboxi(3), rbox(3),vrij(3),fij(3), rij, C_ij(5)
      real*8 p_, ddf1_, df1_,df,dvdr,df_
      real*8 SMALL, SMALLER, phi0, TOLERANCE
      parameter (SMALL=0.001d0, SMALLER=0.00001d0, phi0=0d0)
      parameter (TOLERANCE=0.05d0)

    
      rbox  = (/xbox, ybox, zbox/)
      rboxi = (/xboxi, yboxi, zboxi/)
      do 100 i=1, ntorl
         if(dihtypb(i).ne.2) goto 100
         p=0d0
         fdih1x=0d0
         fdih1z=0d0
         fdih1y=0d0
         fdih2x=0d0
         fdih2z=0d0
         fdih2y=0d0
         fdih3x=0d0
         fdih3z=0d0
         fdih3y=0d0
         fdih4x=0d0
         fdih4z=0d0
         fdih4y=0d0

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
         
         ax = vb1y*vb2zm - vb1z*vb2ym
         ay = vb1z*vb2xm - vb1x*vb2zm
         az = vb1x*vb2ym - vb1y*vb2xm
         bx = vb3y*vb2zm - vb3z*vb2ym
         by = vb3z*vb2xm - vb3x*vb2zm
         bz = vb3x*vb2ym - vb3y*vb2xm

         rasq = ax*ax + ay*ay + az*az
         rbsq = bx*bx + by*by + bz*bz
         rgsq = vb2xm*vb2xm + vb2ym*vb2ym + vb2zm*vb2zm
         rg = sqrt(rgsq)

         rginv = 0.d0
         ra2inv = 0.d0
         rb2inv = 0.d0
         
         if (rg > 0d0) rginv = 1.d0/rg
         if (rasq > 0d0) ra2inv = 1.d0/rasq
         if (rbsq > 0d0) rb2inv = 1.d0/rbsq
         rabinv = sqrt(ra2inv*rb2inv)

         c = (ax*bx + ay*by + az*bz)*rabinv
         s = rg*rabinv*(ax*vb3x + ay*vb3y + az*vb3z)


         if (c > 1.d0)  c =  1.d0
         if (c < -1.d0) c = -1.d0


         
         K_i   = (/rb0(i),rb1(i),rb2(i),rb3(i),rb4(i)/)
         d_phi = (/rbb1(i),rbb2(i),rbb3(i),rbb4(i),rbb5(i)/)
         
         theta = atan2(s, c)
         
         

         p  = 0d0
         df = 0d0
         do jj = 0, 4
            p  = p + K_i(jj+1)*(1d0 + cos(jj*theta - d_phi(jj+1)))
            df = df - K_i(jj+1)*jj*sin(jj*theta - d_phi(jj+1))
         enddo
         df=-df


C     Part with radial potential
C     ij= j - i or from i to j
         vrij = (/-vb1x+vb2x+vb3x,-vb1y+vb2y+vb3y,-vb1z+vb2z+vb3z/)
         rij  = NORM2(vrij)
C         write(*,*) rij
C     Factor to multiply
         fij  = -vrij/rij

C         C_ij = (/rbbb0(i),rbbb1(i),rbbb2(i),rbbb3(i),rbbb4(i)/)

         fij  = rbbb1(i)*(rij-p)*fij
         df   = -df*rbbb1(i)*(rij-p)
C     Adding propensity
         
         
         

C
C         if(mod(istep,100)==0) write(*,*) theta*180.0/3.141592, rij,  p 

         p = 0.5*rbbb1(i)*(rij - p)**2
         
C         write(*,*) fij
C     df=df*POW_COS(rij,C_ij)
C         fij  = fij*D_POW_COS(rij,C_ij)*p
C         p=p*POW_COS(rij,C_ij)
         p_  = 0d0
         df_ = 0d0
         do jj = 0, 4
            p_  = p_ + k1_(jj+1)*(1d0 + cos(jj*theta - k2_(jj+1)))
            df_ = df_ - k1_(jj+1)*jj*sin(jj*theta - k2_(jj+1))
         enddo
         df_=-df_
C         if(mod(istep,100)==0) write (*,*) theta, p_, df_
C         write(*,*) k1_(:)
         df=df+df_
         p=p+p_



         fg = vb1x*vb2xm + vb1y*vb2ym + vb1z*vb2zm
         hg = vb3x*vb2xm + vb3y*vb2ym + vb3z*vb2zm
         fga = fg*ra2inv*rginv
         hgb = hg*rb2inv*rginv
         gaa = -ra2inv*rg
         gbb = rb2inv*rg

         dtfx = gaa*ax
         dtfy = gaa*ay
         dtfz = gaa*az
         dtgx = fga*ax - hgb*bx
         dtgy = fga*ay - hgb*by
         dtgz = fga*az - hgb*bz
         dthx = gbb*bx
         dthy = gbb*by
         dthz = gbb*bz

         sx2 = df*dtgx
         sy2 = df*dtgy
         sz2 = df*dtgz

         fdih1x = df*dtfx
         fdih1y = df*dtfy
         fdih1z = df*dtfz

         fdih2x = sx2 - fdih1x
         fdih2y = sy2 - fdih1y
         fdih2z = sz2 - fdih1z

         fdih4x = df*dthx
         fdih4y = df*dthy
         fdih4z = df*dthz

         fdih3x = -sx2 - fdih4x
         fdih3y = -sy2 - fdih4y
         fdih3z = -sz2 - fdih4z


C

          
         fxatom(ia) = fxatom(ia) + fdih1x - fij(1)
         fyatom(ia) = fyatom(ia) + fdih1y - fij(2)
         fzatom(ia) = fzatom(ia) + fdih1z - fij(3)
         
         fxatom(ib) = fxatom(ib) + fdih2x
         fyatom(ib) = fyatom(ib) + fdih2y
         fzatom(ib) = fzatom(ib) + fdih2z
         
         fxatom(ic) = fxatom(ic) + fdih3x
         fyatom(ic) = fyatom(ic) + fdih3y
         fzatom(ic) = fzatom(ic) + fdih3z
         
         fxatom(id) = fxatom(id) + fdih4x + fij(1)
         fyatom(id) = fyatom(id) + fdih4y + fij(2)
         fzatom(id) = fzatom(id) + fdih4z + fij(3)
         
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
