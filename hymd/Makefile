EXT_SUFFIX := $(shell python3-config --extension-suffix)

all: compute_angle_forces$(EXT_SUFFIX) compute_bond_forces$(EXT_SUFFIX)


compute_angle_forces$(EXT_SUFFIX): compute_angle_forces.f90
	f2py3 -c compute_angle_forces.f90 -m compute_angle_forces

compute_bond_forces$(EXT_SUFFIX): compute_bond_forces.f90
	f2py3 -c compute_bond_forces.f90 -m compute_bond_forces

clean:
	$(RM) *$(EXT_SUFFIX)
