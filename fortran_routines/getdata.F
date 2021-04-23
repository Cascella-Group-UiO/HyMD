      subroutine getdata

      use nlstuff

c     start_doc
c     Name
c       getdata
c     Usage
c       call getdata
c     Description
c       read in from input files
c       title, rcut, dt, nstep, temp0,
c       natom
c       skips comment lines starting with #
c*******************************************
c       elseif (line(1:1) .eq. '#') then
c       goto 1 
c******************************************* 

c     then writes the header of the output file
c     Imported
c     
c     Author 
c     PM april 07 Modified/Simplified from YASP
c     end_doc 

      implicit none
      include 'system.inc'
      include 'iochan.h'
      include 'control.inc'
      integer itemp0, itcoupl, ipress0, ipcoupl, inatom, irfdiel 
      integer ircut, idt, instep, iexcl_type, inlrcut, idiel, ineb
      integer icompr,itranon,ipbctraj,iveltraj,isigma
      integer*8 nstep2
      write (io7, *) '"Frustra fit per plura quod potest  ' 
      write (io7, *) 'fieri per pauciora"'                  
      write (io7, *) ' from                              ' 
      write (io7, *) 'Summa Totius Logicae, Pars I De Terminis       ' 
      write (io7, *) ' Caput 12 Quid est intentio prima et quid secunda' 
      write (io7, *) ' et quomodo distinguuntur ab invicem ' 
      write (io7, *)                                             
      write (io7, *) ' OCCAM MD Run        :'
      write (io7, *) 
      write (io7, *) '  read in input data    ******          '
   
      inatom = 0
      itemp0 = 0
      itcoupl = 0
      ipress0 = 0
      ipcoupl = 0
      iensemble = 0
      ivel_read = 0
      ircut=0
      inlrcut=0
      idt=0
      instep=0
      iexcl_type=0
      idiel=0
      irfdiel=0
      ineb=0
      itrj=0
      iout=0
      icompr=0
      iscf=0
      ipfef=0
      isor=0
      iprotein=0
      imfield=0
      inlsiz = 0 
      itranon = 0 
      ipbctraj = 0 
      iveltraj = 0
      ikappa=0
      isigma=0
      iproelec=0
      i_el_ext=0
C     Default values    

c     dielectric constant and RF dielectric constant

      dielec = 1.0d00
      rfdiel = 1.0d00
c     isothermal compressibility
c     in 1/kPa  for water at 27 C.
      compr  = 4.44d-7

c     default value zero for Lorentz-Berthelot mixing rule  
      imixrul=0
c     default value for nlist updating
      inl=10
c     default value for trajectory updating
      inw=100 
c     default value for output printing     
      ino=100
c     default value for density update on lattice points in SCF simulations
      idl = 10
c     default value for number of accumulated confs. between two updates for density average
      idl2 = 100 
c     Frequency of potential calculation 
      ipot = 10 
c     default value for angle function                                      
      angfun=1
c     default value for connection generation 
      ilis=1

1     read (io1, '(a80)', end = 100, 
     $err = 900) line


c     title
      if (line(1:6) .eq. 'title:') then
         read(io1,*) title
         goto 1
c     cutoff distance
      elseif (line(1:7) .eq. 'cutoff:') then
         read (io1, *) rcut
         ircut=1
         goto 1
      
c     neighbor list cutoff 
      elseif (line(1:10) .eq. 'nl_cutoff:') then
         read (io1, *) nlrcut
         inlrcut=1
         goto 1
c     connectivity generation options
      elseif (line(1:14) .eq. 'conn_lis_mode:') then
c     if ilis 1 bond and angle list are generated from configuration file
c     if ilis 0 bond and angle list are read from fort.12                  
         read (io1, *) ilis 
         goto 1

c     neighbor list length 
      elseif (line(1:8) .eq. 'nl_size:') then
         read (io1, *) nl_size
         inlsiz=1
         goto 1
 
c     time step
      elseif (line(1:10) .eq. 'time_step:') then
         read (io1, *) dt
         idt=1
         goto 1

c     no of iterations
      elseif (line(1:16) .eq. 'number_of_steps:') then
         read (io1, *) nstep2
         nstep=INT(nstep2)
         instep=1
         goto 1

c     neighbour list update frequency
      elseif (line(1:22) .eq. 'update_neighbour_list:') then
         read (io1, *) inl
         ineb=1
         goto 1

c     trajectory     update frequency
      elseif (line(1:10) .eq. 'trj_print:') then
         read (io1, *) inw
         itrj=1
         goto 1

c     angle functional form               
      elseif (line(1:15) .eq. 'angle_function:') then
         read (io1, *) angfun
         goto 1

c     output printing  frequency
      elseif (line(1:10) .eq. 'out_print:') then
         read (io1, *) ino
         iout=1
         goto 1

c     density at lattice point and its derivatives update frequency  SCF simulations
      elseif (line(1:19) .eq. 'SCF_lattice_update:') then
         read (io1, *) idl
         goto 1

C    number of accumulated confs. between two updates for density average
      elseif (line(1:15) .eq.'num_config_acc:') then 
         read(io1, *) idl2
         goto 1

C    Frequency of potential calculation 
      elseif (line(1:14) .eq.'pot_calc_freq:') then
         read(io1, *) ipot
         goto 1

c     ensemble              
      elseif (line(1:19) .eq. 'simulated_ensemble:') then
         read (io1, *) ensemble

         if(ensemble.eq.'NVE') then
           iensemble=1
         endif

         if(ensemble.eq.'NVT') then
           iensemble=2
         endif

         if(ensemble.eq.'NPT') then
           iensemble=3
         endif

         if(ensemble.eq.'NVT_Andersen') then
           iensemble=4
         endif

         goto 1
      
c     target  temperature
      elseif (line(1:19) .eq. 'target_temperature:') then
         read (io1, *) temp0 , tinc
         itemp0=1
         goto 1

c ... dielectric constant
      elseif (line(1:20) .eq. 'dielectric_constant:') then
         read (io1, *) dielec
         idiel=1
         goto 1

c ... dielectric constant for reaction field
      elseif (line(1:26) .eq. 'reaction_field_dielectric:') then
         read (io1, *) rfdiel
         irfdiel=1
         goto 1
      
c     temperature coupling 
      elseif (line(1:18) .eq. 'temperature_coupl:') then
         read (io1, *) tcoupl
         itcoupl=1
         goto 1

c     collision frequency Andersen Thermostat
      elseif (line(1:18) .eq. 'collision_frequen:') then
         read (io1, *) coll_freq
         goto 1

c     target pressure        
      elseif (line(1:16) .eq. 'target_pressure:') then
         read (io1, *) press0
         ipress0=1
         goto 1

c     pressure coupling       
      elseif (line(1:18) .eq. 'pressure_coupling:') then
         read (io1, *) pcoupl
         ipcoupl=1
         goto 1

c     isotermal compressibility 
      elseif (line(1:18) . eq. 'isotherm_compress:') then
         read (io1, *) compr
         icompr=1
         goto 1        
 
c     velocity mode             
      elseif (line(1:14) .eq. 'velocity_read:') then
         read (io1, *) vel_read

         if(vel_read.eq.'yes') then
           ivel_read=1
         endif

         if(vel_read.eq.'no') then
           ivel_read=0
         endif

         goto 1

c     exclusions             
      elseif (line(1:15) .eq. 'exclusion_type:') then
         read (io1, *) excl_type
         iexcl_type = 1 
         goto 1

c     mixing rule            
      elseif (line(1:12) .eq. 'mixing_rule:') then
         read (io1, *) imixrul
         goto 1

      elseif (line(1:12) .eq. 'unlike_fact:') then
         read (io1, *) scun
         imixrul=1
         goto 1
c     number of atoms
      elseif (line(1:6) .eq. 'atoms:') then
         read (io1, *) natom
         inatom=1
         goto 1
c     scf option
      elseif (line(1:11) .eq. 'mean_field:') then
         iscf=1
         read (io1, *) scf_mode 
         if(scf_mode.eq.'SCMF'.or.scf_mode.eq.'PPPM') then
         imfield=1
         endif
         goto 1
!bindu 02/05/2016

        elseif(line(1:25).eq.'Electrostatic_mean_field:') then
        ipfef = 1
        read(io1,*)pfef_mode
        if(pfef_mode.eq.'SOR') then
        isor = 1
        endif
        goto 1

!sigbjorn 28/11/2017

        elseif(line(1:19).eq.'PROTEIN_SIMULATION:') then
        read(io1,*)iprotein
        goto 1
        
      elseif(line(1:12).eq.'PROTEIN_ELEC:') then
         read(io1,*)iproelec
         goto 1
   
      elseif(line(1:5).eq.'E_EXT') then
         i_el_ext=1
         read(io1,*) E_field
         goto 1
   


!ying 20/04/2011

      elseif (line(1:15).eq.'intra_nonbonded') then
        read(io1,*) tranon
        itranon = 1
      goto 1

!ying 20/07/2011

      elseif (line(1:8).eq.'pbc_traj')  then
        read(io1,*) pbctraj
        ipbctraj = 1

      elseif (line(1:13).eq.'velocity_traj') then
        read(io1,*) veltraj
        iveltraj = 1
      goto 1

      elseif (line(1:5).eq.'alpha') then
        read(io1,*) kappa
        ikappa = 1
      goto 1

      elseif (line(1:5).eq.'sigma') then
        read(io1,*) sigmac
        isigma = 1
      goto 1


c     comment line     
      elseif (line(1:1) .eq. '#') then
      goto 1 
c     stop
      elseif (line(1:9) .eq. 'endinput:') then

c     unrecognised line
      else
         write(io7,*) 'FATAL (getdata)'
         write(io7,*) 'bad input line'
         write(io7,*) line                                      
         istop=1
         stop
      endif

c     check minimal information in input file

      if (iensemble.eq.0)  then
         write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
         write(io7,*) ' no ensemble has been set '
         istop=1
      endif

      if (iscf.eq.1.and.imfield.eq.0)  then
         write(io7,*) ' *************************************'
         write(io7,*) ' MEAN FIELD FATAL WARNING!!!!!!!!!!!!!'
         write(io7,*) ' MEAN FIELD FATAL WARNING!!!!!!!!!!!!!'
         write(io7,*) ' Mean Field Approximation has been set'
         write(io7,*) ' But method       ', scf_mode 
         write(io7,*) ' Does not correspond to SCFM or PPPM ' 
         write(io7,*) ' Check Keyword Spelling ' 
         write(io7,*) ' *************************************'
         istop=1
      endif

      if (inatom.eq.0)  then
         write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
         write(io7,*) ' number of atoms not set in input'
         istop=1
      endif
 
      if (idt.eq.0)  then
         write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
         write(io7,*) ' no timestep has been set '
         istop=1
      endif

      if (instep.eq.0)  then
         write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
         write(io7,*) ' no number of step has been set'
         istop=1
      endif
 
      if (ircut.eq.0)  then
         write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
         write(io7,*) ' no cut off radius has been set'
         istop=1
      endif
      
      if (inlrcut.eq.0)  then
         write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
         write(io7,*) ' no nl cut off has been set'
         istop=1
      endif

      if (idiel.eq.0)  then
         write(io7,*) ' DEFAULT                             '
         write(io7,*) ' Dielectric Constant has not been set'
         write(io7,*) ' I use default value = ', dielec
         write(io7,*) 
      endif

      if (irfdiel.eq.0)  then
         write(io7,*) ' DEFAULT                                '
         write(io7,*) ' RF dielectric Constant has not been set'
         write(io7,*) ' I use default value = ', rfdiel
         write(io7,*) 
      endif

      if (ineb.eq.0)  then
         write(io7,*) ' DEFAULT                              '
         write(io7,*) ' nlist updating freq. has not been set'
         write(io7,*) ' I use default value = ', inl         
         write(io7,*) 
      endif

      if (itrj.eq.0)  then
         write(io7,*) ' DEFAULT                              '
         write(io7,*) ' trajectory prn freq. has not been set'
         write(io7,*) ' I use default value = ', inw         
         write(io7,*) 
      endif

      if (icompr.eq.0)  then
         write(io7,*) ' DEFAULT                              '
         write(io7,*) ' isothermal compressibility has not been set'
         write(io7,*) ' I use default value = ', inw         
         write(io7,*) 
      endif

      if (ivel_read.eq.1)  then
         write(io7,*) ' *************************************'
         write(io7,*) ' VELOCITIES                           '
         write(io7,*) ' velocities read from configuration file '
         write(io7,*) '                                         '
      endif

      if (ivel_read.eq.0)  then
         write(io7,*) ' *************************************'
         write(io7,*) ' VELOCITIES not provided/disregarded  '
         write(io7,*) ' velocities initialized in MDINIT     '
         write(io7,*) ' according to T =  ', temp0       
      endif

 100  continue

c     square cut-off radius and nl cutoff radius
      rcutsq=rcut*rcut
      rcutcb=rcutsq*rcut
      RLSTSQ=nlrcut*nlrcut
      xrcut=1.0d00 / rcut
      xrc6=xrcut**6
      xrc12=xrc6**2

c     Reaction field constants
      crf= (rfdiel-1.00d00)/(2.0d00*rfdiel+1.00d00)
      crfrc= crf/rcutcb
      crfrc2=2.0d00*crfrc
      crflrc=(1.0d00+crf)*xrcut

c     timestep stuff
      dt = dt * tconv
      DT2   = DT / 2.0d0
      DTSQ2 = DT * DT2

c     number of freedom
      nfree = 3 * natom 
c ***  input echo

c     beta  1/RT
      beta=1.0/(gascon*temp0)

      write (io7, *) 'title: ', title
      write (io7, *) 'number of atoms:          ',
     $               natom
      write (io7, *) 'cutoff distance:         ', 
     $               rcut
      write (io7, *) 'neighbor list cut_off    ', 
     $               nlrcut
      write (io7, *) 'Dielectric Constant      ', 
     $               dielec
      write (io7, *) 'RF Dielectric Constant      ', 
     $               rfdiel
      write (io7, *) 'time step length:                         ', dt
      write (io7, *) 'number of time steps:                     ',
     $                nstep
      write (io7, *)
      if(angfun.eq.1) then
        write (io7, *) '_--------------------------------------------_'
        write(io7,*) 'ANGLE POTENTIAL harmonic in theta    '
        write (io7, *) '_--------------------------------------------_'
      endif
      if(angfun.eq.2) then
        write (io7, *) '_--------------------------------------------_'
        write(io7,*) 'ANGLE POTENTIAL harmonic in COS(theta)    '
        write (io7, *) '_--------------------------------------------_'
      endif

      if(iscf.eq.1) then
        write (io7, *) '_--------------------------------------------_'
        write(io7,*) 'SIMULATION RUNNING IN MEAN FIELD APPROXIMATION'
        write(io7,*) 'you set  ', scf_mode, ' method'
      
       if(scf_mode.eq.'SCMF') then

        write (io7, *) ' Intermolecular Interactions are replaced by _'
        write (io7, *) '  Density Mean Field                         _'
        write (io7, *) '_--------------------------------------------_'

       endif

       if(scf_mode.eq.'PPPM') then

        write (io7, *)'ALL Interactions at distance larger than cut-off'
        write (io7, *) '  radius  = ', rcut                           
        write (io7, *) '  are replaced by Density Mean Field         _'
        write (io7, *) '_--------------------------------------------_'

       endif

      endif

       if(pfef_mode.eq.'PFEF') then

        write (io7, *) ' Electrostatic Interactions are computed by _'
        write (io7, *) '  computed using Ewald sum                 _'
        write (io7, *) '_--------------------------------------------_'

        elseif(pfef_mode.eq.'SOR')then

        write (io7, *) ' Electrostatic Interactions are computed by _'
        write (io7, *) '  computed using SOR                  _'
        write (io7, *) '_--------------------------------------------_'


       endif


c     Check input consistency

      if(ensemble.eq.'NVE') then
         write (io7, *) '_--------------------------------------------_'
         write (io7, *) '            OCCAM NVE Simulation  '
         write (io7, *) '_--------------------------------------------_'
      endif
  
      if(ensemble.eq.'NVT') then
         write (io7, *) '_--------------------------------------------_'
         write (io7, *) '            OCCAM NVT Simulation  '
         write (io7, *) '_--------------------------------------------_'
         if(itemp0.eq.0) then
           write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
           write(io7,*) ' you set NVT ensemble but not target T!!!'
           istop=1
         endif
         if(itcoupl.eq.0) then
           write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
           write(io7,*) ' you set NVT ensemble but not T coupling !!'
           istop=1
         endif
         write (io7, *) 'Target temperature:         ',
     $                  temp0
         write (io7, *) 'Temperature coupling is: ',
     $                  tcoupl
      endif

      if(ensemble.eq.'NPT') then
         write (io7, *) '_--------------------------------------------_'
         write (io7, *) '            OCCAM NPT Simulation  '
         write (io7, *) '_--------------------------------------------_'
         if(ipress0.eq.0) then
           write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
           write(io7,*) ' you set Pcontrol on but not target P!!!'
           istop=1
         endif
         if(ipcoupl.eq.0) then
           write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
           write(io7,*) ' you set pcontrol on but not P coupling !!'
           istop=1
         endif
         if(itemp0.eq.0) then
           write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
           write(io7,*) ' you set NPT ensemble but not target T!!!'
           istop=1
         endif
         if(itcoupl.eq.0) then
           write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
           write(io7,*) ' you set NPT ensemble but not T coupling !!'
           istop=1
         endif
         write (io7, *) 'Target temperature:         ',
     $                  temp0
         write (io7, *) 'Temperature coupling is: ',
     $                  tcoupl
         write (io7, *) 'Target pressure:         ',
     $                  press0
         write (io7, *) 'Pressure coupling is: ',
     $                  pcoupl
      endif

c     check exclusion settings

      if(iexcl_type.eq.0) then
        write(io7,*) ''
        write(io7,*) '******************************'
        write(io7,*) 'WARNING exclusion type not set'
        write(io7,*) 'I feel free to assign default 1_3'
        excl_type='1_3'
      endif

      if(iexcl_type.eq.1) then
        if(excl_type.eq.'1_2'.or
     $    .excl_type.eq.'1_3'.or
     $    .excl_type.eq.'1_4'.or
     $    .excl_type.eq.'no_intra'.or
     $    .excl_type.eq.'read') then
          write(io7,*) 'exclusions type:   ', excl_type
        else
          write(io7,*) 'FATAL WARNING!!   ', excl_type, 'exclusion type 
     $ unknown'
        istop=1
        endif
      endif

      if (ivel_read.eq.0.and.itemp0.eq.0)  then
         write(io7,*) 'FATAL WARNING'
         write(io7,*) ' VELOCITIES not provided/disregarded  '
         write(io7,*) ' SET INITIAL TEMPERATURE OR PROVIDE VELOCITIES'
      endif

      if(iscf.eq.1) then
        if(mod(idl, idl2).ne.0) then
           write(io7,*) ' FATAL WARNING!!!!!!!!!!!!!'
           write(io7,*) 'num_config_acc: is ', idl2
           write(io7,*) 'num_config_acc: is not a submultiple of', idl
           istop=1
           else
           write(io7,*) 'Density AVERAGING'
           write(io7,*) 'For Density averaging a config every ',idl/idl2
           write(io7,*) 'steps will be used                         '
           write(io7,*) 'for a total of ', idl2, ' confs'
         endif
       endif

      return

 900  write (io7, *) '*** can''t read input file'
      istop=1
      end 

