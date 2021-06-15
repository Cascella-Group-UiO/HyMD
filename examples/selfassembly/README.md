The steps we follow to see the self-assembly of DPPC in water. Let us use as an example 8 DPPC in water:
1. Use CHARMM-GUI to make Martini structure of DPPC in water.  
2. Download the `.tgz`, unzip, and run Gromacs for preparing the system. See `job_charmm_equil.sh`
3. Convert `.gro` (output from Gromacs at the end of previous step) to `fort.5` and then to `.h5`.
4. See `job_destroy_bilayer.sh`.  It does the following:
   i) Switch off field (use `disable-field` flag) . It raises the thermostat temperature to 500K. Thermostat coupling groups only between individual cpl groups. Run for 1ns. Use `utils/h5md2input.py` to extract last frame for next step. Call it `hymd_inp_1.h5`
   ii) Switch on only compressibility term (`hamiltonian = DefaultNoChi`. Remove `disable-field` flag). Still at 500K. Thermostat between all cpl + water groups. Run for 1ns. Use `utils/h5md2input.py` to extract last frame for next step. Call it `hymd_inp_2.h5`
   iii) Bring down temperature to 323K. No other configuration changes. Run for 1ns. Use `utils/h5md2input.py` to extract last frame for output. Call it `hymd_inp_3.h5`
   
   
