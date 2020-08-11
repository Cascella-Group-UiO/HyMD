# Hamiltonian and Alias-Free Hybrid Particle-Field Molecular Dynamics MPI VERSION
This depository contains a simple python-code implementation of hybrid particle-field molecular dynamics (hPF-MD) that allows one to conserve energy and reduce aliasing by refinement of grid. The code supports monoatom particles of any amount of types and interactions.

The code is contained in main.py and input files are created with make_input.py from CONF.py. To convert results to unitless units the units.py module is provided.

To try out the code, simply:
   python make_input.py CONF.py 
   mpirun python main.py CONF.py input.hdf5

Note that to run the code, the following may need to be installed:
pip3 install pmesh
pip3 install pfft-python

# MPI configured h5py

The following procedure worked to install mpi configured h5py on a macbook pro running macOS Catalina 10.15.3

brew --version
>>> Homebrew 2.4.9
brew install python@3.8
python3 --version
>>> Python 3.8.5
pip3 --version
>>> pip 20.1.1 from usr/local/lib/python3.8/site-packages/pip (python 3.8 )
brew install open-mpi
brew install hdf5-mpi
brew install pgk-config
pip3 install mpi4py pmesh numpy sympy pfft-python mpsort  cython 

Using LLVM default compiler from Xcode
mpicc --showme
>>> clang -I/usr/local/Cellar/open-mpi/4.0.4_1/include -L/usr/local/opt/libevent/lib -L/usr/local/Cellar/open-mpi/4.0.4_1/lib -lmpi

compile h5py from source
git clone git@github.com:h5py.git
cd h5py

master is probably fine, but I used this specific one
The lastest github release 2.10.0 from Sep 2019 does *not* work
git checkout 6f4c578f78321b857da31eee0ce8d9b1ba291888
HDF5_MPI="ON" pip3 install -v .