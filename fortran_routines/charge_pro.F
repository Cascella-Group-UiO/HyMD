      subroutine charge_pro
      use scfstuff
      use nlstuff
      implicit none
      include 'iochan.h'
      include 'system.inc'
      include 'config.inc'
      include 'forces.inc'
      include 'control.inc'
      include 'lists.inc'
      include 'model.inc'

C     !!!!!IMPORTANTE!!!!!!!!!!!!!!!!!!!
C      aggiungere parte di codice che copia negli elementi vuoti
C      i valori di densità dettati dalle PBCC
C      Ma forse non serve ???? controllare
C      FATTO Serve per evitare di usare sempre la funzione PBCIND 
C      che puo' rallentare l'esecuzione

!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!
C, add some code to copy in the empty elements
C density values dictated by PBCC
C But maybe not needed??? control
c Is made is to avoid always using the function PBCIND
C which can 'slow down the execution

      integer i, j, k, icell, ibeg, iend, ipr
      integer v1, v2, v3, v4, pbcindP 
      integer v5, v6, v7, v8 , iadd, jadd

!     real*8 lx1, ly1, lz1
      real*8 xj, yj, zj, c_dtot, conv
      
C     Subroutine for calculating forces due to protein backbone
          
      integer  ia, ib, ic, id, ii, jj, kk
      integer  jb, jbeg, jend, NNN, index
     
      real*8 vb2x,vb2y,vb2z,vb3x,vb3y
      real*8 vb3z,vb2xm,vb2ym,vb2zm
      real*8 b2mag2, b2mag, b3mag2, b3mag
      real*8 calpha, invsalpha, alpha, delta
      real*8 cpsi, spsi,qval,qr
      real*8 theta, ctheta, stheta, dtheta
      real*8 rpp, rnp, rpn, rnn,r_pos(3),r_neg(3)
      real*8 vrnn(3), vrnp(3),vrpn(3),vrpp(3),xr,yr,zr
      
      real*8 force_fac, p, fac
 
 
     
      real*8 v_1(3),v_2(3), vn(3), vm(3)
      real*8 r0(3), d2(3), rboxi(3), rbox(3), vr_center(3)
     $     ,r_center
      parameter(delta=0.3)


      cpsi = 0.177              !cos(1.77)
      spsi = sqrt(1-cpsi**2)    !sin(1.77)

      qval= (0.020819434*3.6/delta)*sqrt(qfact2)
      

      r_neg = 0.0d0
      r_pos = 0.0d0
      p     = 0.0d0
      fac   = 0.0d0

      c_dtot=0d0

      
      rbox  = (/xbox, ybox, zbox/)
      rboxi = (/xboxi, yboxi, zboxi/)
C     Loop for reconstructing dipoles with single charges
      
      do 100  i=1, nangl 
         if(angltypb(i).ne.2) goto 100

         ib = liang(i)
         ic = ljang(i)
         id = lkang(i)
         
         vb2x = xpbc(ic) - xpbc(ib)
         vb2y = ypbc(ic) - ypbc(ib)
         vb2z = zpbc(ic) - zpbc(ib)
         
         vb3x = xpbc(id) - xpbc(ic)
         vb3y = ypbc(id) - ypbc(ic)
         vb3z = zpbc(id) - zpbc(ic)

         if(vb2x>0.5*xbox) vb2x=vb2x-xbox
         if(vb3x>0.5*xbox) vb3x=vb3x-xbox
         if(vb2y>0.5*ybox) vb2y=vb2y-ybox
         if(vb3y>0.5*ybox) vb3y=vb3y-ybox
         if(vb2z>0.5*zbox) vb2z=vb2z-zbox
         if(vb3z>0.5*zbox) vb3z=vb3z-zbox
         

         if(vb2x<-0.5*xbox) vb2x=vb2x+xbox
         if(vb3x<-0.5*xbox) vb3x=vb3x+xbox
         if(vb2y<-0.5*ybox) vb2y=vb2y+ybox
         if(vb3y<-0.5*ybox) vb3y=vb3y+ybox
         if(vb2z<-0.5*zbox) vb2z=vb2z+zbox
         if(vb3z<-0.5*zbox) vb3z=vb3z+zbox
c$$$
c$$$         
c$$$         vb2x = vb2x - xbox*anint(xboxi * vb2x)
c$$$         vb3x = vb3x - xbox*anint(xboxi * vb3x)         
c$$$         
c$$$         vb2y = vb2y - ybox*anint(yboxi * vb2y)
c$$$         vb3y = vb3y - ybox*anint(yboxi * vb3y)
c$$$         
c$$$         vb2z = vb2z - zbox*anint(zboxi * vb2z)
c$$$         vb3z = vb3z - zbox*anint(zboxi * vb3z)
c$$$         
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
          
         ctheta = cos(theta)
         stheta = sin(theta)
         
         v_1 = (/vb2xm, vb2ym, vb2zm/)/b2mag 
         v_2 = (/vb3x, vb3y, vb3z/)/b3mag
         
         vn = invsalpha*CROSS(v_1, v_2)
         vm = CROSS(vn, v_2)
         

         r0 = (/xpbc(ic), ypbc(ic), zpbc(ic)/)
     $        + 0.5d0 * (/vb3x, vb3y, vb3z/) 
         d2  = 0.50d0*delta*(cpsi*v_2 + spsi*(ctheta*vn + stheta*vm))
         
         r_pos = r0 + d2
         r_neg = r0 - d2
         
C     Taking care of periodicity
c$$$         r_pos = r_pos - rbox * anint (rboxi * r_pos)
c$$$         r_neg = r_neg - rbox * anint (rboxi * r_neg)
         
        


         

         
         do k=1,2
            if(k.eq.1)then
               xr=r_pos(1)
               yr=r_pos(2)
               zr=r_pos(3)
               qr=qval
            else
               xr=r_neg(1)
               yr=r_neg(2)
               zr=r_neg(3)
               qr=-qval
            endif

            if(xr<0d0)then
               xr=xr+xbox
            elseif(xr>xbox)then
               xr=xr-xbox
            endif

            if(yr<0d0)then
               yr=yr+ybox
            elseif(yr>ybox)then
               yr=yr-ybox
            endif

            if(zr<0d0)then
               zr=zr+zbox
            elseif(zr>zbox)then
               zr=zr-zbox
            endif
c$$$
c$$$            write(*,*) xr,yr,zr
c$$$            xr=xr - xbox * anint(xr * xboxi)
c$$$            yr=yr - ybox * anint(yr * yboxi)
c$$$            zr=zr - zbox * anint(zr * zboxi)
c$$$            write(*,*) xr,yr,zr

            Icell=  1+ (int(xr/ lx ))
     $             + (int(yr/ ly ))*mx
     $             + (int(zr/ lz ))*mxy
      

          v1   =  v(1,icell)
          v2   =  v(2,icell)
          v3   =  v(3,icell)
          v4   =  v(4,icell)
          v5   =  v(5,icell)
          v6   =  v(6,icell)
          v7   =  v(7,icell)
          v8   =  v(8,icell)
          
          
          xj=xr-lx1*(int(xr*ilx1))
          yj=xr-ly1*(int(yr*ily1))
          zj=xr-lz1*(int(zr*ilz1))
 
          
          c_d(v1) = c_d(v1)+(lx1-xj)*(ly1-yj)*(lz1-zj)*ivcell*qr
          c_d(v2) = c_d(v2)+xj*(ly1-yj)*(lz1-zj)*ivcell*qr
          c_d(v3) = c_d(v3)+(lx1-xj)*yj*(lz1-zj)*ivcell*qr
          c_d(v4) = c_d(v4)+xj*yj*(lz1-zj)*ivcell*qr
          c_d(v5) = c_d(v5)+(lx1-xj)*(ly1-yj)*zj*ivcell *qr
          c_d(v6) = c_d(v6)+xj*(ly1-yj)*zj*ivcell*qr
          c_d(v7) = c_d(v7)+(lx1-xj)*yj*zj*ivcell*qr
          c_d(v8) = c_d(v8)+xj*yj*zj*ivcell*qr

       enddo
 100  continue
                    
         

       do i=1,nvert
          c_dtot=c_dtot+c_d(i)
       enddo

       do j = 1 , mz    
          jadd=(j-1)*nlay
          
          do i = 1 , ncol-1
             iadd=i*nrow+jadd
             
             c_d(iadd)= c_d( pbcindP(iadd))
             
             
          enddo   
          
       enddo
       
       do j = 1, nfloor-1
          jadd=(j-1)*nlay
          do i=1,nrow
             
             c_d(nlay-nrow+i+jadd)
     $            = c_d(pbcindP(nlay-nrow+i+jadd))
 

          enddo
          
       enddo 
            
       do i = 1, nlay
          c_d(nvert-nlay+i)=c_d(i)
          
       enddo
            
!       write(io7,*)'total charge density',c_dtot




      return 
      

      contains
      FUNCTION CROSS(a,b)
C     Crossproduct between two vectors
      IMPLICIT NONE
      REAL*8 CROSS(3)
      REAL*8, INTENT(IN):: a(3), b(3)
      
      CROSS(1) = a(2)*b(3) - a(3)*b(2)
      CROSS(2) = a(3)*b(1) - a(1)*b(3)
      CROSS(3) = a(1)*b(2) - a(2)*b(1)
      END FUNCTION CROSS
      end
      function pbcindP(iin)
      use scfstuff
      integer pbcindP
      
      
      if(iin.le.0) then
         iin=iin+nrow-1
      endif
C     row number
      nper=1+int((iin-1)/nrow)
      
C     periodicity along one row
      it=iin-(nrow-1)*int(iin/(nper*nrow))
      
C     periodicity along one column
      it2= it-(ncol*nrow-nrow)*int((it+nrow-1)/(nrow*ncol))
      
      it3= it2+(nlay-nrow)*int((iin-1)/(nlay))

      pbcindP = it3-nlay*(nfloor-1)*int((iin-1)/(nlay*(nfloor-1)))
      
      end
