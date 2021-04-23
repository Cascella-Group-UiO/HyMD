      subroutine fdihedral   
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
      real*8 d_phi(8),K_i(8)
      real*8 ax,ay,az,bx,by,bz,rasq,rbsq,rgsq,rg
      real*8 rginv,ra2inv,rb2inv,rabinv,c,s,theta
      real*8 fg,hg,fga,hgb,gaa,gbb,dtfx,dtfy,dtfz,dtgx,dtgy,dtgz
      real*8 dthx,dthy,dthz,sx2,sy2,sz2,df,p

      real*8 p_, ddf1_, df1_      
      real*8 SMALL, SMALLER, phi0, TOLERANCE
      parameter (SMALL=0.001d0, SMALLER=0.00001d0, phi0=0d0)
      parameter (TOLERANCE=0.05d0)
      do 100 i=1, ntorl
         if(dihtypb(i).ne.1.and.dihtypb(i).ne.7.and.dihtypb(i).ne.6)
     $        goto 100

         p=0d0
         ia = litor(i)
         ib = ljtor(i)
         ic = lktor(i)
         id = lltor(i)

C        Define vectors between particles
         vb1x =  -(x(ib) - x(ia))
         vb1y =  -(y(ib) - y(ia))
         vb1z =  -(z(ib) - z(ia))

         vb2x =  x(ic) - x(ib)
         vb2y =  y(ic) - y(ib)
         vb2z =  z(ic) - z(ib)
         
         vb3x =  x(id) - x(ic)
         vb3y =  y(id) - y(ic)
         vb3z =  z(id) - z(ic)
         
C         Is this even necessary?         
C         vb2xm = -vb2x
C         vb2ym = -vb2y
C         vb2zm = -vb2z
         
C        PBC 
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
C        Doesn't make much sense to take the reverse of both vb1 and vb2
C        (vb2m)

C        Cross product components
C        vb1 x vb2m
         ax = vb1y*vb2zm - vb1z*vb2ym
         ay = vb1z*vb2xm - vb1x*vb2zm
         az = vb1x*vb2ym - vb1y*vb2xm

C        vb3 x vb2m
         bx = vb3y*vb2zm - vb3z*vb2ym
         by = vb3z*vb2xm - vb3x*vb2zm
         bz = vb3x*vb2ym - vb3y*vb2xm

C        Norms
         rasq = ax*ax + ay*ay + az*az
         rbsq = bx*bx + by*by + bz*bz

C        The norm is the same for the non-reversed vector vb2
         rgsq = vb2xm*vb2xm + vb2ym*vb2ym + vb2zm*vb2zm
         rg = sqrt(rgsq)

C        Inverses to normalize dot products
         rginv = 0.d0
         ra2inv = 0.d0
         rb2inv = 0.d0
         
         if (rg > 0d0) rginv = 1.d0/rg
         if (rasq > 0d0) ra2inv = 1.d0/rasq
         if (rbsq > 0d0) rb2inv = 1.d0/rbsq
         rabinv = sqrt(ra2inv*rb2inv)

C        Dot product
         c = (ax*bx + ay*by + az*bz)*rabinv
         s = rg*rabinv*(ax*vb3x + ay*vb3y + az*vb3z)


         if (c > 1.d0)  c =  1.d0
         if (c < -1.d0) c = -1.d0


         
        
         K_i   = (/rb0(i),rb1(i),rb2(i),rb3(i),rb4(i),rb5(i),rb6(i)
     $        ,rb7(i)/)
         d_phi = (/rbb1(i),rbb2(i),rbb3(i),
     $        rbb4(i),rbb5(i),rbb6(i),rbb7(i),rbb8(i)/)
         theta = atan2(s, c)
         p  = 0d0
         df = 0d0
         do jj = 0, 7
            p  = p + K_i(jj+1)*(1d0 + cos(jj*theta - d_phi(jj+1)))
            df = df - K_i(jj+1)*jj*sin(jj*theta - d_phi(jj+1))
         enddo
         df=-df

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
         
         fxatom(ia) = fxatom(ia) + fdih1x
         fyatom(ia) = fyatom(ia) + fdih1y
         fzatom(ia) = fzatom(ia) + fdih1z
         
         fxatom(ib) = fxatom(ib) + fdih2x
         fyatom(ib) = fyatom(ib) + fdih2y
         fzatom(ib) = fzatom(ib) + fdih2z
         
         fxatom(ic) = fxatom(ic) + fdih3x
         fyatom(ic) = fyatom(ic) + fdih3y
         fzatom(ic) = fzatom(ic) + fdih3z
         
         fxatom(id) = fxatom(id) + fdih4x
         fyatom(id) = fyatom(id) + fdih4y
         fzatom(id) = fzatom(id) + fdih4z

C         write(*,*) i,fdih1x,fdih2x,fdih3x
         vtors = vtors + p

 100  CONTINUE
      
      return


      end
