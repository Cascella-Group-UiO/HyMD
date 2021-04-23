      subroutine fparklevitt
C     Subroutine for calculating forces due to protein backbone

      implicit none
      include 'system.inc'
      include 'control.inc'
      include 'iochan.h'
      include 'lists.inc'
      include 'config.inc'
      include 'model.inc'
      include 'forces.inc'
      
      integer i, ia, ib, ic, id,jj
      real*8 fdih1x, fdih1y, fdih1z, fdih2x, fdih2y, fdih2z
      real*8 fdih3x, fdih3y, fdih3z, fdih4x, fdih4y, fdih4z
      real*8 vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y
      real*8 vb3z,vb2xm,vb2ym,vb2zm
      real*8 d_phi(5),K_i(5),l1_(5),l2_(5),gamma0_,dgamma0_
      real*8 ax,ay,az,bx,by,bz,rasq,rbsq,rgsq,rg
      real*8 rginv,ra2inv,rb2inv,rabinv,c,s,phi
      real*8 fg,hg,fga,hgb,gaa,gbb,dtfx,dtfy,dtfz,dtgx,dtgy,dtgz
      real*8 dthx,dthy,dthz,sx2,sy2,sz2,p
      real*8 rboxi(3), rbox(3)
      real*8 df,df_,k_const,gamma0, dk_const
      real*8 calpha,b1mag,b2mag,salpha,a,a11,a12,b1mag2,b2mag2,a22
      real*8 fang1x1,fang1y1,fang1z1,fang2x1,fang2y1,fang2z1
      real*8 fang3x1,fang3y1,fang3z1,gamma
      real*8 fang2x2,fang2y2,fang2z2
      real*8 fang3x2,fang3y2,fang3z2
      real*8 fang4x2,fang4y2,fang4z2
      real*8 SMALL, SMALLER, TOLERANCE,df3
      real*8 b3mag2,b3mag,salpha1,calpha1,gamma1
      parameter (SMALL=0.001d0, SMALLER=0.00001d0)
      parameter (TOLERANCE=0.05d0)
    
    
      rbox  = (/xbox, ybox, zbox/)
      rboxi = (/xboxi, yboxi, zboxi/)
      do 100 i=1, ntorl
         if(dihtypb(i).ne.7) goto 100
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

         fang1x1=0d0         
         fang1y1=0d0
         fang1z1=0d0

         fang2x1=0d0         
         fang2y1=0d0
         fang2z1=0d0

         fang3x1=0d0         
         fang3y1=0d0
         fang3z1=0d0

         fang2x2=0d0         
         fang2y2=0d0
         fang2z2=0d0

         fang3x2=0d0         
         fang3y2=0d0
         fang3z2=0d0
         
         fang4x2=0d0         
         fang4y2=0d0
         fang4z2=0d0

         gamma0_=0d0
         dgamma0_=0d0
         k1_=(/kb1(i),kb2(i),kb3(i),kb4(i),kb5(i)/)
         k2_=(/kdb1(i),kdb2(i),kdb3(i),kdb4(i),kdb5(i)/)

         l1_=(/lb1(i),lb2(i),lb3(i),lb4(i),lb5(i)/)
         l2_=(/ldb1(i),ldb2(i),ldb3(i),ldb4(i),ldb5(i)/)

         p =0d0
         
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

         phi = atan2(s, c)
         
C     Angles
         b1mag2  = vb1x*vb1x+vb1y*vb1y+vb1z*vb1z
         b2mag2  = vb2x*vb2x+vb2y*vb2y+vb2z*vb2z
         b1mag   = sqrt(b1mag2)
         b2mag   = sqrt(b2mag2)
         calpha  = vb1x*vb2x + vb1y*vb2y + vb1z*vb2z
         calpha  = calpha/(b1mag*b2mag)
        
         if (calpha >  1.0d0) calpha =  1.d0
         if (calpha < -1.0d0) calpha = -1.d0
        
         salpha = sqrt(1.d0 - calpha*calpha)
         if (salpha < SMALL) salpha = SMALL
         salpha = 1.d0/salpha
         gamma  = acos(calpha)

         if(int(rbbb1(i))==1) then
            b3mag2  = vb3x*vb3x+vb3y*vb3y+vb3z*vb3z
            
            b3mag   = sqrt(b3mag2)
            
            calpha1  = vb3x*vb2xm + vb3y*vb2ym + vb3z*vb2zm
            calpha1  = calpha1/(b2mag*b3mag)
            
            if (calpha1 >  1.0d0) calpha1 =  1.d0
            if (calpha1 < -1.0d0) calpha1 = -1.d0
            
            salpha1 = sqrt(1d0 - calpha1*calpha1)
            if (salpha1 < SMALL) salpha1 = SMALL
            salpha1 = 1.d0/salpha1
            gamma1  = acos(calpha1)
         endif


         k_const  = 0d0
         dk_const = 0d0
         do jj = 0, 4
            k_const  = k_const +
     $           k1_(jj+1)*(1d0 + cos(jj*phi - k2_(jj+1)))
            dk_const = dk_const -
     $           k1_(jj+1)*jj*sin(jj*phi - k2_(jj+1))

            gamma0_  = gamma0_ +
     $           l1_(jj+1)*(1d0 + cos(jj*phi - l2_(jj+1)))
            dgamma0_ = dgamma0_ -
     $           l1_(jj+1)*jj*sin(jj*phi - l2_(jj+1))
         enddo
         
C     Removed 2
c$$$
         k_const=k_const 
         dk_const=dk_const
         



         
C     V(phi,gamma) = 0.5*(gamma-gamm0(phi))^2
C     gamma0(phi)  = 106-13*cos(gamma-45)
C     gamma0(phi)  = 1.85 - 0.227*cos(gamma - 0.785)
C         dgamma = 0.227*sin(gamma-0.785)
C         gamma0 = 1.85d0 - 0.227d0*cos(phi-0.785d0)
         df     = -k_const*dgamma0_*(gamma-gamma0_)
     $        + 0.5d0*dk_const*(gamma-gamma0_)**2
         
         df     = -df 
        
         df_    = k_const*(gamma - gamma0_)
         
         if(int(rbbb1(i))==1) then
            df3 = k_const*(gamma1 - gamma0_)
            df  = df + k_const*dgamma0_*(gamma1-gamma0_)
     $           - 0.5d0*dk_const*(gamma1 - gamma0_)**2
            p = p + 0.5d0*k_const*(gamma1 - gamma0_)**2
         endif
C     if(mod(istep,100)==0) write(*,*)phi,k_const
         
C Dihedral part        
        fg  = vb1x*vb2xm + vb1y*vb2ym + vb1z*vb2zm
        hg  = vb3x*vb2xm + vb3y*vb2ym + vb3z*vb2zm
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

C     Angles
        a   = -df_ * salpha
        a11 = a*calpha / b1mag2
        a12 = -a / (b1mag*b2mag)
        a22 = a*calpha / b2mag2
        
        
        
        fang1x1 = a11*vb1x + a12*vb2x
        fang1y1 = a11*vb1y + a12*vb2y
        fang1z1 = a11*vb1z + a12*vb2z
        
        fang3x1 = a22*vb2x + a12*vb1x
        fang3y1 = a22*vb2y + a12*vb1y
        fang3z1 = a22*vb2z + a12*vb1z         
        
        fang2x1 = -(fang1x1 + fang3x1)
        fang2y1 = -(fang1y1 + fang3y1)
        fang2z1 = -(fang1z1 + fang3z1)
        
C     Second angle
        if(int(rbbb1(i)) == 1) then
           a   = -df3 * salpha1
           a11 = a*calpha1 / b2mag2
           a12 = -a / (b2mag*b3mag)
           a22 = a*calpha1 / b3mag2
           
           fang2x2 = a11*vb2xm + a12*vb3x
           fang2y2 = a11*vb2ym + a12*vb3y
           fang2z2 = a11*vb2zm + a12*vb3z
           
           fang4x2 = a22*vb3x + a12*vb2xm
           fang4y2 = a22*vb3y + a12*vb2ym
           fang4z2 = a22*vb3z + a12*vb2zm         

           fang3x2 = -(fang2x2 + fang4x2)
           fang3y2 = -(fang2y2 + fang4y2)
           fang3z2 = -(fang2z2 + fang4z2)
        endif
        
C

        

        
        fxatom(ia) = fxatom(ia) + fdih1x + fang1x1 
        fyatom(ia) = fyatom(ia) + fdih1y + fang1y1
        fzatom(ia) = fzatom(ia) + fdih1z + fang1z1
        
        fxatom(ib) = fxatom(ib) + fdih2x + fang2x1 + fang2x2
        fyatom(ib) = fyatom(ib) + fdih2y + fang2y1 + fang2y2
        fzatom(ib) = fzatom(ib) + fdih2z + fang2z1 + fang2z2
        
        fxatom(ic) = fxatom(ic) + fdih3x + fang3x1 + fang3x2
        fyatom(ic) = fyatom(ic) + fdih3y + fang3y1 + fang3y2
        fzatom(ic) = fzatom(ic) + fdih3z + fang3z1 + fang3z2
        
        fxatom(id) = fxatom(id) + fdih4x + fang4x2
        fyatom(id) = fyatom(id) + fdih4y + fang4y2
        fzatom(id) = fzatom(id) + fdih4z + fang4z2
        
        vtors = vtors + 0.5d0*k_const*(gamma-gamma0_)**2 + p
 100  CONTINUE
      
      return
      end
