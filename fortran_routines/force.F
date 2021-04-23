      subroutine force         
      implicit none
      include 'system.inc' 
      include 'iochan.h'
      include 'config.inc' 
      include 'model.inc'
      include 'forces.inc'

      integer i

c     write(io7,*) 'START FORCE'      
 
 
c     initialization of forces and energies

      do i = 1, natom
         fxatom(i) = 0.0d00
         fyatom(i) = 0.0d00
         fzatom(i) = 0.0d00
      enddo

      do i = 1, 6
         pvinb(i)  = 0.0d00
         pvibnd(i) = 0.0d00
         pviang(i) = 0.0d00
         pvicos(i) = 0.0d00
         pvilin(i) = 0.0d00
         pvitor(i) = 0.0d00
         pvihd(i)  = 0.0d00
      enddo 

      vbond      = 0.0d00
      vangle     = 0.0d00
      vcosa      = 0.0d00
      vlina      = 0.0d00
      vtors      = 0.0d00
      vhd        = 0.0d00
      vnb        = 0.0d00
      vnbns      = 0.0d00
      vscf1      = 0.0d00
      vscf2      = 0.0d00

c      write(io7,*) 'ENERGY CONTRIBUTIONS'
c      write(io7,*) 'INIT                '
 
c      write(io7,*)'vbond = ', vbond    
c      write(io7,*)'vangle = ',vangle   
c      write(io7,*)'vcosa = ', vcosa    
c      write(io7,*)'vlina = ', vlina    
c      write(io7,*)'vtors = ', vtors    
c      write(io7,*)'vhd = ', vhd      
c      write(io7,*)'vnb = ',  vnb      
c      write(io7,*)'vnbns = ', vnbns    
  

c$$$C     Reconstruction of side chains
      call reconstruct_cb

c     bond stretching forces, if any
      

      if (nbtype.gt.0) then
         call fbond 
      endif

c     angle bending forces, if any

      if (nangtype.gt.0.and.angfun.eq.1) then
         call fangle
      endif

      if (nangtype.gt.0.and.angfun.eq.2) then
         call fangle_cos 
      endif

      if ((tranon.eq.1).and.(nbtype.gt.0)) then

         call fnb
      endif

      if (nttype.gt.0) then
         call fdihedral
         call fdihedral_special
C         call fdihedral_vdv
         
C         call fbackbone 
         call fparklevitt
         if(iproelec) call f_intra_electro
      endif


        if(ipfef.eq.1) then
          call f_elec
          if(isor.eq.1) call f_pol
          if(iprotein.eq.1) call comb
          
       endif



      if (scf_mode.eq.'SCMF') then
         call f_mfield
      endif


      

!      if (scf_mode.eq.'PPPM') then
!         call f_mfield2
!      endif
c$$$
      call fside_chain
c$$$      if(mod(istep,inw).eq.0)then
c$$$      do i = 1, natom
c$$$         write(*,*) i,fxatom(i),fyatom(i),fzatom(i)
c$$$      enddo
c$$$      endif

      return
      
      end
