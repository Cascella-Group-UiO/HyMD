#!/bin/bash
#SBATCH --job-name=pmesh
#SBATCH --account=nn4654k
#SBATCH --time=0-0:10:0
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=1G
#SBATCH --qos=devel

#set -o errexit # exit on errors
module load h5py/2.10.0-foss-2020a-Python-3.8.2
module load pfft-python/0.1.21-foss-2020a-Python-3.8.2

cp ../hPF_MD_PMESH/{make_input.py,main.py} .
bash ../hPF_MD_PMESH/ex.sh

python gprof2dot.py -f pstats cpu_0.prof | dot -Tpng -o profile_graph.png
#mpirun python3  ../hPF_MD_PMESH/main.py CONF.py input.hdf5 
