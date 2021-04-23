C     System.Inc



C     Information About The System
C
C
C     character*80 title  input title
C     character*80 blank  blank
C     character*80 line, excl_type   
C     real*8 rcut, rcutsq         cut-off radius
C     real*8 nlrcut, RLSTSQ       neighbor list  cut-off radius
C     real*8 dt           timestep
C     real*8 temp0        initial temperature
C     integer nstep       number of steps
C     data blank / ' ' /

      character*40 title, ensemble,vel_read,scf_mode,pfef_mode
      character*15 excl_type            
      character*80 line 
      character*80 blank
      real*8 rcut, rcutsq, nlrcut, rlstsq, xrcut
      real*8 dt, crf, crfrc, crfrc2, xrc6, xrc12, crflrc
      real*8 dt2, dtsq2, vol, virlrc,kappa,sigmac,fact

      
 
      real*8 temp0, avsnum, degrad, tcoupl, press0, pcoupl
      real*8 tinc, scun, coll_freq, beta,e_field
      real*8 rfdiel, dielec, rcutcb , tke, temp, vscal, press 
      real*8 press_pp, press_pf0, press_pf1, press_pf1x 
      real*8 press_pf1y, press_pf1z 
      real*8 tpress_pp, tpress_pf0, tpress_pf1, tpress_pf1x 
      real*8 tpress_pf1y, tpress_pf1z, tpress 
      real*8 ttemp, tbond, tangle, tnb 
      real*8 weight, dens, pke(6), compr, xscal 
      integer nstep, natom, istep, nfree, ntype, icount
      data blank / ' ' /

 
C     vectors and matrixes sizes
      integer maxnat, maxnbn, maxnum, maxnb 
      integer maxpnt, maxtor, maxang , maxbntyp  
      integer maxtyp, maxlbn, nbondm, iensemble, inlsiz, ivel_read                       
      integer imixrul, inl, inw, itrj, iout, ino, idp, idl, iscf,
     & ipfef,ikappa,isor,iprotein,iproelec,i_el_ext
c ying 9/9/2010
      integer idl2,ipot, ilis         
c ying 20/04/2011
      integer tranon
!ying 20/07/2011

      integer pbctraj, veltraj                                                

C     ierr 
      integer ierr, imfield, angfun

      Common /Sys1/    rcut, dt, temp0, press0, tcoupl, 
     $pcoupl, rlstsq, rcutsq, nlrcut, rfdiel, dielec,
     $crf, crfrc, crfrc2, rcutcb, xrc6, xrc12, xrcut, crflrc, 
     $dt2, dtsq2, tke, temp,  vol, weight, dens, press,
     $ press_pp, press_pf0, press_pf1, press_pf1x, press_pf1y,
     $ press_pf1z, tpress,  
     $ tpress_pp, tpress_pf0, tpress_pf1, tpress_pf1x, tpress_pf1y,
     $ tpress_pf1z, 
     $ pke, compr, xscal, vscal, virlrc, tinc, ttemp, 
     $ tbond, tangle, tnb, scun, coll_freq, beta,kappa,sigmac,fact
     
      Common /Sys2/    nstep, natom, iensemble, inlsiz, imixrul, 
     $istep, inl, inw, itrj, nfree, ivel_read, iout, ino, idp, idl, 
     $iscf, imfield, ntype, icount, angfun, idl2,ipot,tranon,
     $pbctraj,veltraj, ilis,ipfef,ikappa,isor,iprotein,iproelec,
     $ i_el_ext,E_field

      Common /Sys3/    title, line
      Common /Sys4/    excl_type,ensemble,vel_read,
     $scf_mode,pfef_mode

C     vectors and matrixes sizes
      parameter (maxnat = 500000)   ! max number of atoms
C     maxnbn must be equal to the number of connectivity atoms
      parameter (maxnbn = 6) ! max num of bonded atom 
      parameter (maxnum = 500000) ! max number of molecules
      parameter (maxnb = 30)   ! max number of nonbonded types
      parameter (maxpnt = 500) ! max number of points in tables
      parameter (maxtor = 40 ) ! max number of torsions        
      parameter (maxang = 40 ) ! max number of angles          
      parameter (maxbntyp = 30)! max number of bond types  
      parameter (maxtyp = 35)! max number of atom types     
      parameter (maxlbn=maxnat*maxnbn)! max no.of bonds in the system
      parameter (nbondm=2*maxlbn)! length of bond list vectors
C     physical constants
      parameter (avsnum = 6.0221367d+23) ! Avogadro Number
C     conversion factors
      parameter (degrad = 0.017453d00)  !  degs to radiants 

c     unit conversion factors

c     pi (CRC Handbook)
      real*8 pi
      parameter (pi = 3.14159 26535 89793 23846 26433 8d00)

c     1 atomic mass unit = 1.6605655(86) e-27 kg (CRC Handbook 1982)
      real*8 amu
      parameter (amu = 1.6605655d-27)

c     universal gas constant R = 8.3143 e-3 kJ / (K mole)
c     (CRC Handbook 1982)
      real*8 gascon
      parameter (gascon = 8.3143d-3)
      
c     Avogadro's number Navogadro = 6.022045 e23 1 / mole
c     accurate to 1 % (CRC Handbook 1982)
      real*8	avogad
      parameter (avogad = 6.022045d23)

c     Boltzmann's constant k = R / Navogadro  kJ / K 
      real*8	boltz
      parameter (boltz = gascon / avogad)

c     rescaling of time step so that velocities are in nm/ps
c     (forces in kJ/(mole * nm), time in ps, masses in amu, 
c     distances in nm). should equal 1, ideally.
      
      real*8	tconv
      parameter (tconv = 1.0d0)

c     vacuum permittivity (CRC Handbook)
c     epsilon0 = 8.85418782(7) e-12 [C**2 / (J m) = C / (V m)]
      real*8 vacper
      parameter (vacper = 8.85418782d-12)

c     elementary charge: e = 1.6021892(46) e-19 C (CRC Handbook)
      real*8 	elcha
      parameter (elcha = 1.6021892d-19)

c     epsilon0 in local units [(e**2 mol) / (kJ nm)]
      real*8	epsi0
      parameter (epsi0 = vacper / (avogad * elcha**2 * 1.0d06))

c     factor for scaling charges so that q(1) * q(2) / r in
c     [e**2/nm] gives an energy in kJ/mol
      real*8	qfact2
      parameter (qfact2 = 1.0d00 / (4.0d00 * pi * epsi0))
c     this is actually the square of the conversion factor
c     since Fortran 77 does not like functions in parameter
c     statements
      real*8 t1,t2,t3,t4
      common/tt/t1,t2,t3,t4
