# Hamiltonian and Alias-Free Hybrid Particle-Field Molecular Dynamics MPI VERSION
This depository contains a simple python-code implementation of hybrid particle-field molecular dynamics (hPF-MD) that allows one to conserve energy and reduce aliasing by refinement of grid. The code supports monoatom particles of any amount of types and interactions.

The code is contained in main.py and input files are created with make_input.py from CONF.py. To convert results to unitless units the units.py module is provided.

To try out the code, simply:
   python make_input.py CONF.py 
   mpirun python main.py CONF.py input.hdf5

Note that to run the code, the following may need to be installed:
pip3 install pmesh
pip3 install pfft-python
