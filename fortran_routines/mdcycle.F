      subroutine mdcycle
      implicit none
      include 'system.inc' 
      include 'iochan.h' 
      include 'config.inc'
      include 'forces.inc'
      include 'control.inc'
     
      real*8 tscal, pscal
      integer i
      integer idl2_step

      idl2_step=idl/idl2

c     box length inversion

      xbox = b(1,1)
      xboxi  = 1.0d00 / xbox
      ybox = b(2,2)
      yboxi  = 1.0d00 / ybox
      zbox = b(3,3)
      zboxi  = 1.0d00 / zbox
      vol = b(1,1)*b(2,2)*b(3,3)

      call mdinit

      call pbc 

      if (tranon.eq.1) then 

      call nlist
 
      endif 

      if(iscf.eq.1) then

        call dens_cell

        call gfield5 

        call dens_xyz

      endif 



        if(ipfef.eq.1)then
           call chrg_dens_cell
           if(iprotein.eq.1) then
              call charge_pro
           endif
           
        if(isor.eq.1) then
        call eps_dens_cell
        call sor
        
C     

      else
        call fft_d
        endif

C        call SAVE_FIELD
        call chrg_gfield5
C     Added for external electrostatic field
        if(i_el_ext.eq.1)then
           call chrg_ext 
        endif
        call chrg_dens_xyz
C     Sigbjorn 04/04/2018
        if(isor.eq.1)then
           call P_cell              
           call eps_grad
           call P_gfield5
           call pol_dens_xyz
        endif
C        
        endif

   
      call force

      call cinet

C     Main MD Loop

C     initialization of properties averages 

C     Pressure

      tpress         = 0.00              
      tpress_pp      = 0.00       
      press_pf0     = 0.00     
      tpress_pf1     = 0.00 
      tpress_pf1x    = 0.00 
      tpress_pf1y    = 0.00
      tpress_pf1z    = 0.00 

      write(io7,*) 'MDCYCLE'
 
      call CPU_TIME(t3)

      do istep=1,nstep

        if (istep.eq.1.or.mod(istep,inl).eq.0) then
  
         b(1,1)=b(1,1)+binc 
         b(2,2)=b(2,2)+binc 
         b(3,3)=b(3,3)+binc 
  
        endif

        if ((tranon.eq.1).and.(istep.eq.1.or.mod(istep,inl).eq.0)) then
 
             call nlist
 
        endif

         call integrate
        
         call pbc 

      if (iscf.eq.1) then

         if(mod(istep,idl2_step).eq.0) then

            call dens_acc

         endif

      
         if (mod(istep,idl).eq.0) then

            call dens_upd
      
            call gfield5

         endif
      
       
            call dens_xyz

      endif

         if(ipfef.eq.1) then

c         if(mod(istep,idl2_step).eq.0) then
c         call chrg_dens_acc
c         endif

         if (mod(istep,idl).eq.0) then

c        call chrg_dens_upd

        call chrg_dens_cell
        if(iprotein.eq.1) then
           call charge_pro
        endif
        if(isor.eq.1) then
         call eps_dens_cell
         call sor
         else
         call fft_d
         endif
         call chrg_gfield5
C     Added for external electrostatic field
         if(i_el_ext.eq.1) then
            call chrg_ext 
         endif
         if(isor.eq.1)then
           call P_cell              
           call eps_grad
           call P_gfield5
        endif
        

         
      endif
      
         call chrg_dens_xyz
         
         if(isor.eq.1) call pol_dens_xyz
         
      endif

      
         call force
         
         call velgen

c     box scaling factor

        if ( iensemble .eq.  3  ) then

         call bscal

        endif

c     velocity scaling factor

        if ( iensemble .eq. 2 .or. iensemble. eq. 3  ) then
      
             temp0=temp0+tinc
       
             vscal = tscal (temp, temp0, tcoupl, dt)

        else
 
             vscal = 1.0d00
     
        endif

        
        if ( iensemble .eq. 4  ) then

            temp0=temp0+tinc

            call andersen                            

        endif

          call cinet

     
        if (istep.eq.1.or.mod(istep,ino).eq.0) then

          call values

          call pvalues

        endif

        if (istep.eq.1.or.mod(istep,inw).eq.0) then
 
          call wtrj    
  
        endif

        enddo 
   
        return
        end
