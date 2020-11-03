EXT_SUFFIX := $(shell python3-config --extension-suffix)
F90FLAGS = "-Ofast -pedantic"
FC := f2py3 --verbose --f90flags=${F90FLAGS}

all: \
	compute_angle_forces$(EXT_SUFFIX)  \
	compute_bond_forces$(EXT_SUFFIX)   \
	compute_gaussian_core$(EXT_SUFFIX) \


compute_angle_forces$(EXT_SUFFIX): compute_angle_forces.f90
	$(FC) -c compute_angle_forces.f90 -m compute_angle_forces

compute_bond_forces$(EXT_SUFFIX): compute_bond_forces.f90
	$(FC) -c compute_bond_forces.f90 -m compute_bond_forces

compute_gaussian_core$(EXT_SUFFIX): compute_gaussian_core.f90
	$(FC) -c compute_gaussian_core.f90 -m compute_gaussian_core

clean:
	$(RM) *$(EXT_SUFFIX)
