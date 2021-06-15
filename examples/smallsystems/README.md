HyMD uses Martini coarse-graining mapping. A Martini structure can be created in many ways. Here we use CHARMM-GUI to build it and then convert it into the `.h5` format that HyMD needs.
Let's use as an example a very small system of a DPPC bilayer with 4DPPC in a box of size `[1.58750, 1.58750, 10.0]` nm.  
Start here: `CHARMM-GUI` > `Input Generator` > `Martini Maker` > `Bilayer Builder`  
We will build a `Martini only system`  
`Select Martini Models:` Select `martini22`  
Follow the instructions. Note that distance units is `Å`.  
Download and unzip the `.tgz` file to get folder `charmm-gui-XXXXXXXXXX`. This folder contains a `gromacs` folder with all necessary files you need to run Gromacs simulations (`.pdb`, `.mdp`, `.top`, etc.) It also contains a `toppar` folder with the necessary forcefield (Martini) files.

#### `.pdb` to `.gro` without Gromacs run
The main file we need is the `pdb` file. It is usually called `step5_charmm2gmx.pdb`.  
__Way 1:__ If you don't care about a gromacs pre-run before switching to HyMD format, this is the only file you need.  
One of the many ways to convert `.pdb` to `.gro` :  
`vmd step5_charmm2gmx.pdb`  
Right-click on the file in `vmd` and click `save coordinates`. Select `.gro` format and write `all`.
Name the file `step5_charmm2gmx.gro`

__Way 2:__ Use `gmx pdb2gmx -f step5_charmm2gmx.pdb -o step5_charmm2gmx.gro`. Make sure you use the correct forcefields. All relevant forcefields should be in the folder `toppar` though some Gromacs versions might cause some errors and require some additional files (like `.atp`)

#### Do a Gromacs run and get a `.gro`
__Way 3__:
