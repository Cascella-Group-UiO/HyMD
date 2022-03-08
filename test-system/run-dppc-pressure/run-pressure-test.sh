## test for Lasse GPE

mpirun -n 1 python3 -m  hymd configtest_HyMD.tomli dppc-with-charge-dielectric.h5  --double-precision --verbose  --velocity-output  --logfile GPE-log.txt
