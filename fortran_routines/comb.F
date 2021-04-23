      subroutine comb
C     Subroutine for calculating forces due to protein backbone
      use nlstuff
      use scfstuff
      implicit none

      include 'system.inc'
      include 'control.inc'

      include 'lists.inc'
      include 'config.inc'
      include 'model.inc'
      include 'forces.inc'
  
      include 'iochan.h'
      

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

      integer k, icell, ibeg, iend, ipr
      integer k1, k2, k3, k4, pbcindPro 
      integer k5, k6, k7, k8 , iadd, jadd

!     real*8 lx1, ly1, lz1
      real*8 xj, yj, zj, c_dtot, conv
      real*8 xr,zr,yr,qr,qval
      real*8 cgdx1, cgdy1, cgdz1
      real*8 dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8
      real*8 dv10, dv20, dv30, dv40, dv50, dv60, dv70, dv80

      real*8 temp1,temp2,temp3,temp4,temp5
      ! force 
      integer i, ia, ib, ic, id, ii, jj, kk
      integer j, jb, jbeg, jend, NNN, index


!      REAL*8 OUTER_PRODUCT(3,3),CROSS_MATRIX(3,3),cross(3)
     
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
      real*8 r0(3), d1(3), rboxi(3), rbox(3), f2(3), vr_center(3)
     $     ,r_center
      parameter(delta=0.3)
      cpsi = 0.177              !cos(1.77)
      spsi = sqrt(1-cpsi**2)    !sin(1.77)


      
      qval= (0.020819434*3.6/delta)*sqrt(qfact2)

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
         
         vb2x = xpbc(ic) - xpbc(ib)
         vb2y = ypbc(ic) - ypbc(ib)
         vb2z = zpbc(ic) - zpbc(ib)
         
         vb3x = xpbc(id) - xpbc(ic)
         vb3y = ypbc(id) - ypbc(ic)
         vb3z = zpbc(id) - zpbc(ic)


     
         if(vb2x>0.5*xbox) vb2x=vb2x-xbox
         if(vb3x>0.5*xbox) vb3x=vb3x-xbox
         if(vb2x<-0.5*xbox) vb2x=vb2x+xbox
         if(vb3x<-0.5*xbox) vb3x=vb3x+xbox


         if(vb2y>0.5*ybox) vb2y=vb2y-ybox
         if(vb3y>0.5*ybox) vb3y=vb3y-ybox
         if(vb2y<-0.5*ybox) vb2y=vb2y+ybox
         if(vb3y<-0.5*ybox) vb3y=vb3y+ybox


         if(vb2z>0.5*zbox) vb2z=vb2z-zbox
         if(vb3z>0.5*zbox) vb3z=vb3z-zbox
         if(vb2z<-0.5*zbox) vb2z=vb2z+zbox
         if(vb3z<-0.5*zbox) vb3z=vb3z+zbox

        
         
c$$$         vb2x = vb2x - xbox*anint(xboxi * vb2x)
c$$$         vb3x = vb3x - xbox*anint(xboxi * vb3x)         
c$$$         
c$$$         vb2y = vb2y - ybox*anint(yboxi * vb2y)
c$$$         vb3y = vb3y - ybox*anint(yboxi * vb3y)
c$$$         
c$$$         vb2z = vb2z - zbox*anint(zboxi * vb2z)
c$$$         vb3z = vb3z - zbox*anint(zboxi * vb3z)


c$$$         vb2x = vb2x - xbox*anint(xboxi * vb2x)
c$$$         vb3x = vb3x - xbox*anint(xboxi * vb3x)         
c$$$         
c$$$         vb2y = vb2y - ybox*anint(yboxi * vb2y)
c$$$         vb3y = vb3y - ybox*anint(yboxi * vb3y)
c$$$         
c$$$         vb2z = vb2z - zbox*anint(zboxi * vb2z)
c$$$         vb3z = vb3z - zbox*anint(zboxi * vb3z)
         
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
         r0 = (/xpbc(ic), ypbc(ic), zpbc(ic)/) 
     $        + 0.5d0 * (/vb3x, vb3y, vb3z/) 
         d1  = 0.50d0*delta*(cpsi*v2 + spsi*(ctheta*vn + stheta*vm))
         
         r_pos(:,i) = r0 + d1
         r_neg(:,i) = r0 - d1
         
C     Taking care of periodicity
C         r_pos(:,i) = r_pos(:,i) - rbox * anint (rboxi * r_pos(:,i))
C         r_neg(:,i) = r_neg(:,i) - rbox * anint (rboxi * r_neg(:,i))
         
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

               
          do k=1,2
            if(k.eq.1)then
               xr=r_pos(1,i)
               yr=r_pos(2,i)
               zr=r_pos(3,i)
               qr=qval
            else
               xr=r_neg(1,i)
               yr=r_neg(2,i)
               zr=r_neg(3,i)
               qr=-qval
            endif

c$$$
c$$$            xr=xr-xbox*anint(xboxi*xr)
c$$$            yr=yr-ybox*anint(yboxi*yr)
c$$$            zr=zr-zbox*anint(zboxi*zr)
c$$$
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
c$$$            xr=xr - xbox * anint(xr * xboxi)
c$$$            yr=yr - ybox * anint(yr * yboxi)
c$$$            zr=zr - zbox * aninlt(zr * zboxi)
            icell=  1 + (int(xr/ lx))
     $                + (int(yr/ ly))*mx
     $                + (int(zr/ lz))*mxy
      

            xj=xr-lx*(int(xr*ilx))
            yj=yr-ly*(int(yr*ily))
            zj=zr-lz*(int(zr*ilz))


           
         
          temp1 = lx-xj
          temp2 = (ly-yj)*(lz-zj)*ivcell
          temp3 = yj*(lz-zj)*ivcell
          temp4 = (ly-yj)*zj*ivcell
          temp5 = yj*zj*ivcell

          dv10 = temp1*temp2
          dv20 = xj*temp2
          dv30 = temp1*temp3
          dv40 = xj*temp3
          dv50 = temp1*temp4
          dv60 = xj*temp4
          dv70 = temp1*temp5
          dv80 = xj*temp5



         if(mod(istep,idl).eq.0)then                 

         k1   =  feng( v(1,icell) )
         k2   =  feng( v(2,icell) )
         k3   =  feng( v(3,icell) )
         k4   =  feng( v(4,icell) )
         k5   =  feng( v(5,icell) )
         k6   =  feng( v(6,icell) )
         k7   =  feng( v(7,icell) )
         k8   =  feng( v(8,icell) )



           dv1= k1*dv10
           dv2= k2*dv20
           dv3= k3*dv30
           dv4= k4*dv40
           dv5= k5*dv50
           dv6= k6*dv60
           dv7= k7*dv70
           dv8= k8*dv80
       

           tot_ele=tot_ele+qr*(dv1 + dv2 + dv3 + dv4 +
     $          dv5 + dv6 + dv7 + dv8)



         endif
         cgdx1=grad_x(v(1,icell))*dv10+grad_x(v(2,icell))*dv20+
     &             grad_x(v(3,icell))*dv30+grad_x(v(4,icell))*dv40+
     &             grad_x(v(5,icell))*dv50+grad_x(v(6,icell))*dv60+
     &             grad_x(v(7,icell))*dv70+grad_x(v(8,icell))*dv80



       cgdy1=grad_y(v(1,icell))*dv10+grad_y(v(2,icell))*dv20+
     &             grad_y(v(3,icell))*dv30+grad_y(v(4,icell))*dv40+
     &             grad_y(v(5,icell))*dv50+grad_y(v(6,icell))*dv60+
     &             grad_y(v(7,icell))*dv70+grad_y(v(8,icell))*dv80


       cgdz1=grad_z(v(1,icell))*dv10+grad_z(v(2,icell))*dv20+
     &             grad_z(v(3,icell))*dv30+grad_z(v(4,icell))*dv40+
     &             grad_z(v(5,icell))*dv50+grad_z(v(6,icell))*dv60+
     &             grad_z(v(7,icell))*dv70+grad_z(v(8,icell))*dv80

        if(k.eq.1)then
           f_pos(1, i) = -cgdx1*qr
           f_pos(2, i) = -cgdy1*qr
           f_pos(3, i) = -cgdz1*qr
        else
           f_neg(1, i) = -cgdx1*qr
           f_neg(2, i) = -cgdy1*qr
           f_neg(3, i) = -cgdz1*qr
        endif
       enddo

 100  CONTINUE



      
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
      end
      
