EXT_SUFFIX := $(shell python3-config --extension-suffix)
F90FLAGS = "-Ofast -pedantic -ffree-line-length-0"
FC := f2py3 --verbose --f90flags=${F90FLAGS}
SINGLE_TO_DOUBLE := "s/real(4)/real(8)/g"

all: \
	compute_bond_forces__double.f90   				\
	compute_bond_forces__single$(EXT_SUFFIX)   		\
	compute_bond_forces__double$(EXT_SUFFIX)   		\
	compute_angle_forces__double.f90  				\
	compute_angle_forces__single$(EXT_SUFFIX)  		\
	compute_angle_forces__double$(EXT_SUFFIX)  		\
	compute_dihedral_forces__double.f90  			\
	compute_dihedral_forces__single$(EXT_SUFFIX)  	\
	compute_dihedral_forces__double$(EXT_SUFFIX)  	\
	protein_backbone.f90 							\
	protein_backbone_single$(EXT_SUFFIX)  			\
	protein_backbone_double$(EXT_SUFFIX)  			\
	compute_gaussian_core$(EXT_SUFFIX) 				\

compute_gaussian_core$(EXT_SUFFIX): compute_gaussian_core.f90
	$(FC) -c compute_gaussian_core.f90 -m compute_gaussian_core

compute_angle_forces__single$(EXT_SUFFIX): compute_angle_forces.f90
	$(FC) -c compute_angle_forces.f90 -m compute_angle_forces

compute_angle_forces__double.f90: compute_angle_forces.f90
	sed ${SINGLE_TO_DOUBLE} compute_angle_forces.f90 > compute_angle_forces__double.f90

compute_angle_forces__double$(EXT_SUFFIX): compute_angle_forces__double.f90
	$(FC) -c compute_angle_forces__double.f90 -m compute_angle_forces__double
	$(RM) compute_angle_forces__double.f90

compute_bond_forces__single$(EXT_SUFFIX): compute_bond_forces.f90
	$(FC) -c compute_bond_forces.f90 -m compute_bond_forces

compute_bond_forces__double.f90: compute_bond_forces.f90
	sed ${SINGLE_TO_DOUBLE} compute_bond_forces.f90 > compute_bond_forces__double.f90

compute_bond_forces__double$(EXT_SUFFIX): compute_bond_forces__double.f90
	$(FC) -c compute_bond_forces__double.f90 -m compute_bond_forces__double
	$(RM) compute_bond_forces__double.f90

compute_dihedral_forces__single$(EXT_SUFFIX): compute_dihedral_forces.f90
	$(FC) -c compute_dihedral_forces.f90 -m compute_dihedral_forces

compute_dihedral_forces__double.f90: compute_dihedral_forces.f90
	sed ${SINGLE_TO_DOUBLE} compute_dihedral_forces.f90 > compute_dihedral_forces__double.f90

compute_dihedral_forces__double$(EXT_SUFFIX): compute_dihedral_forces__double.f90
	$(FC) -c compute_dihedral_forces__double.f90 -m compute_dihedral_forces__double
	$(RM) compute_dihedral_forces__double.f90

protein_backbone__single$(EXT_SUFFIX): protein_backbone.f90
	$(FC) -c protein_backbone.f90 dipole_reconstruction.f90 -m protein_backbone 

protein_backbone__double.f90: protein_backbone.f90
	sed ${SINGLE_TO_DOUBLE} protein_backbone.f90 > protein_backbone__double.f90
	sed ${SINGLE_TO_DOUBLE} dipole_reconstruction.f90 > dipole_reconstruction__double.f90

protein_backbone__double$(EXT_SUFFIX): protein_backbone__double.f90
	$(FC) -c protein_backbone__double.f90 dipole_reconstruction__double.f90 -m protein_backbone__double
	$(RM) protein_backbone__double.f90

clean:
	$(RM) *$(EXT_SUFFIX)
	$(RM) *__double.f90
	$(RM) -r compute_gaussian_core/
	$(RM) -r compute_bond_forces/
	$(RM) -r compute_angle_forces/
	$(RM) -r compute_dihedral_forces/
	$(RM) -r protein_backbone/
