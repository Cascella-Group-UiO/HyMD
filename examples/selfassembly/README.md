We do NVT runs for self-assembly.

### Self-assembly of lipids to form a bilayer-configuration
The steps we follow to see the self-assembly of DPPC in water.
1. Use CHARMM-GUI to make Martini structure of DPPC in water.  
2. Download the `.tgz`, unzip, and run Gromacs for preparing the system. See `job_charmm_equil.sh`: Input - `step 6.4*` ; Output - `step9_production.*`  
3. We now do an equilibration run in HyMD. See `job_hymd_equil.sh`: Input - `step9_production.gro`; Output - `hymd_equilibrated.h5`
   i) Convert `.gro` (output from Gromacs at the end of previous step) to `fort.5` and then to `.h5`.  
   ii) Anneal (?) through the following steps:  
       (A) 10 K, 1 ns;  
       (B) 100 K, 1 ns;  
       (C) 200 K, 1 ns;   
       (D) 300 K, 1 ns;  
       (E) 323 K, 10 ns  
   iii) Extract last frame. Call it `hymd_equilibrated.h5`.  
4. Now we run HyMD. See `job_destroy_bilayer.sh`.  It does the following:\
   i) Switch off field (use `disable-field` flag) . It raises the thermostat temperature to 500K. Thermostat coupling groups only between individual cpl groups. Run for 1ns. Use `utils/h5md2input.py` to extract last frame for next step. Call it `hymd_inp_1.h5`  
   ii) Switch on only compressibility term (`hamiltonian = DefaultNoChi`. Remove `disable-field` flag). Still at 500K. Thermostat between all cpl + water groups. Run for 1ns. Use `utils/h5md2input.py` to extract last frame for next step. Call it `hymd_inp_2.h5`  
   iii) Bring down temperature to 323K. No other configuration changes. Run for 1ns. Use `utils/h5md2input.py` to extract last frame for output. Call it `hymd_inp_3.h5` that you will find in the folder `hymd-random-out`  
5. `hymd_inp_3.h5` is the input for HyMD. Run a regular job script with this input. For example, you can run `job_self_assembly.sh` which runs the same MD with 4 different seeds.

We expect to see self-assembly.
What did we do: We started from an equilibrated stable bilayer. Blew it up by raising the temperature. Cooled it down. Ran HyMD. We did not start from a random system. This is for being able to compare the kind of self-assembled structure we get from HyMD with the structure we had built on CHARMM-GUI followed by Gromacs equilibration.

### Self-assembly of lipids to form vesicles
__WAY 1:__ We do not start from a structured system. The reason is the self-assembled structure we expect to get is not an equilibrium structure. So there is no need for a mechanism in which we compare the HyMD-self-assembled-structure with a Gromacs-equilibrated-structure. We calculate the ratios of lipids and water and size of box blabla that can possibly aggregate into a vesicle. Then we just throw in those molecules randomly in a box and run HyMD.

__WAY 2:__ We build a vesicle from CHARMM-GUI or Packmol. We do not bother with running Gromacs (i.e. step 2 of bilayer self-assembly) or with HyMD equilibration (i.e. step 3 of bilayer self-assmebly).
1. Use CHARMM-GUI to make Martini structure of DPPC vesicle in water. I obtained a vesicle with `6 pores`.
  > **System Details**  
  >  DPPC  
  > \# lipids Outer layer: 1748  
  >           Inner Layer: 1133  
  > \# water             : 185278  
  > Box: [30.0, 30.0, 30.0]
   
2. Download the `.tgz`, unzip. Find `step5_charmm2gmx.pdb`. Convert to `step5_charmm2gmx.gro` (See ways to do it in [examples/small systems](https://github.com/Cascella-Group-UiO/HyMD-2021/tree/pressure/examples/smallsystems).) Check if this `.gro` has the correct indices and velocities. If yes, jump to step 4.
4. If no velocities are present, simply add `0.000  0.000  0.000` for every particle. If indices are not correct, use:  
   `python3 ~/hymdtest/utils/split_gro_molecules.py step5_charmm2gmx.gro --out step5_charmm2gmx_split.gro -f`  
4. Convert `.gro` to `.h5`:  
   `python3 ~/hymdtest/utils/gro2fort5.py step5_charmm2gmx_split.gro --out step5_charmm2gmx.5 -f`  
   `python3 ~/hymdtest/utils/fort5_to_hdf5.py step5_charmm2gmx.h5 --out charmm_inp.h5`  
5. Blow it up (step 4 of bilayer self-assembly) by:  
   `sbatch job_destroy_bilayer.sh config_destroy_bilayer.toml charmm_inp.h5`
6. Then we run HyMD (step 5 of bilayer self-assembly).
