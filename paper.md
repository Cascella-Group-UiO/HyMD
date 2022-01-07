---
title: 'HylleraasMD: Massively parallel hybrid particle-field molecular dynamics in python'
tags:
  - chemistry
  - physics
  - molecular dynamics
  - coarse-grained
  - hybrid particle-field
authors:
  - name: Morten Ledum^[corresponding author]
    orcid: 0000-0003-4244-4876
    affiliation: 1
  - name: Manuel Carrer
    orcid: 0000-0002-8777-4310
    affiliation: 1
  - name: Samiran Sen
    orcid: 0000-0002-1922-7796
    affiliation: 1
  - name: Xinmeng Li
    orcid: 0000-0002-6863-6078
    affiliation: 1
  - name: Michele Cascella
    orcid: 0000-0003-2266-5399
    affiliation: 1
  - name: Sigbjørn Løland Bore
    orcid: 0000-0002-8620-4885
    affiliation: 2
affiliations:
  - name: Department of Chemistry, and Hylleraas Centre for Quantum Molecular Sciences,University of Oslo, PO Box 1033 Blindern, 0315 Oslo, Norway
    index: 1
  - name: Department of Chemistry and Biochemistry, University of California San Diego, La Jolla,California 92093, United States
    index: 2
date: 7 January 2022
bibliography: paper.bib
---

# Summary

Molecular dynamics (MD) is a computational methodology in which the dynamical behavior of systems of interacting atoms and molecules is investigated by integrating the corresponding classical equations of motion. The analysis of the molecular trajectories yields an incredibly powerful computational microscope with atomic resolution. While prominent examples of molecular dynamics involving all-atom models exist, many systems operate on time- and lengths scales too large, precluding the use of such an approach. The intrinsic complexity of biological soft matter systems has necessitated the development of *coarse-grained* (CG) MD models wherein groups of atoms are treated as individual entities. To probe experimentally relevant length- (nm&ndash;$\mathrm{mu}$m) and time- (ps&ndash;ms) scales, further reduction of computational complexity may be warranted through the removal of explicit particle\textendash particle interactions in favor of particle\textendash density field interactions. Such *hybrid particle&ndash;field* (hPF) models recast the interactions between particle pairs into a system of free particles interacting with an external potential dependent on the density, in analogy with self-consistent field theories.

HylleraasMD (named after our affiliate centre, the *Hylleraas Centre for Quantum Molecular Sciences*) (HyMD) is a python package capable of highly parallel hPF-MD simulations of a wide range of surfactants and other biological systems in a CG representation. At present, it is the only open source implementation of the hPF formalism freely available to computational researchers.

# Theoretical background

Hybrid particle&ndash;field methods are computationally efficient schemes for simulating mesoscale macromolecular assemblies [@Milano:2009]. Ordinary MD involves, at every integration step, the calculation of computationally expensive double sum over all particle pairs. Despite numerous clever decompositions of the simulation box, which reduces the formal scaling of this [@Frenkel:2001], it remains the major computational bottleneck. Hybrid particle&ndash;field simulations forego this step completely, instead indirectly coupling particles only through an *interaction energy functional* depending on a slowly varying density field. Exploiting the slow time evolution of the density fields, it is possible to employ multiple time-step algorithms which only seldom impart field impulses on individual particles. Beyond this fundamentally more efficient setup of hPF models, the major advantage over particle&ndash;particle approaches is the intrinsically *embarrassingly parallel* nature of a large portion of the calculations; inter-MPI communication only being necessary whenever the density field is updated. This is traditionally done every tens&ndash;several hundreds of MD steps. Accordingly, the hPF methodology has been successfully applied to polymer melts [@Wu:2020; @Wu:2021], different phases of lipids and surfactants [@Nicola:2015; @Carrer:2020; @Bore:2020:1; @Nicola:2020; @Bore:2019; @Ledum:2020], and charged surfactants and polypeptides [@Kolli:2018; @Schafer:2020; @Bore:2018].

# Statement of need

Elucidating fundamental aspects of the complexity of biological systems often require atomically resolved mesoscale simulations. One crucial example is the large-scale macromolecular self-assembly of lipids and proteins into eukaryotic cell membranes or intracellular organelle structures. Some such systems are computationally accessible today at the CG-MD level, but this is far from routine and not achievable for the broad scientific community. Hybrid particle&ndash;field models allow in principle exploration of such systems at near-atomistic resolution, with good chemical accuracy.

Since the hPF scheme was proposed [@Milano:2013], two main codes have been used to perform such simulations. (i) OCCAM [@Zhao:2012], a proprietary Fortran software developed by Milano and co-workers; and (ii) GALAMOST [@Zhu:2013], a CUDA-GPU accelerated C++ code developed by researchers at Jilin University. Unfortunately, neither are open source and freely available to scientists wishing to run hPF simulations of bio- and soft-matter systems.

HyMD is, to date, the only available open-source hPF simulation software. Furthermore, through a recent reformulation of the hPF formalism [@Bore:2020:2], which decouples the computational mesh grid and the length scale of the particle\textendash grid interaction, a new *Hamiltonian* hPF (HhPF) method has emerged. Currently, HyMD constitutes the only software for performing HhPF simulations, open-source or otherwise. This new scheme has a number of advantages over canonical hPF, such as rigorous energy and momentum conservation, rotationally and translationally-invariant forces, and a tunable coarse-graining length scale representing the size extent of particles. Additionally, the new formulation naturally lends itself to calculation in reciprocal space, enabling us to take advantage of highly optimized FFT algorithms.

# Features

Apart from a minimal set of high-performance Fortran kernels, the entirety of HyMD is written in python. This makes extending the software with new functionality easy, enabling fast prototyping of new features. The key components of HyMD include:

- Standard hPF interaction functionals, with the option to specify *any* (local or otherwise) functional, which is automatically handled through symbolic differentiation and numpy vectorization.
- Density filtering (with any user-provided filter function), enabling canonical hPF or HhPF simulations with tunable coarse-graining scale which can be changed *on-the-fly*.
- Optional explicit electrostatic interactions through our custom long-range Particle-Mesh Ewald.
- All standard intramolecular bonded interactions, including stretching, bending, torsional potentials, and combined bending&ndash;torsional potentials describing peptide backbone conformations [@Bore:2018].
- Topological reconstruction of permanent peptide chain backbone dipoles, enabling realistic protein conformational simulations [@Cascella:2008; @Alemani:2010; @Bore:2018].

To probe experimentally relevant structures, parallelization through mpi4py is used. A 2D *pencil grid* domain decomposition is employed, separating spatial areas of the simulation box across MPI ranks. The highly scalable PFFT [@Pippig:2013] library is used for reciprocal space calculations, as a backend for the PMESH [@Feng:2017] package through which we handle the particle&ndash;mesh part of the code. A specialized HDF5 file format for MD trajectories [@Buyl:2014] is used to enable massively parallel file IO while maintaining an easy structural organization of quantities calculated for storage.

# Availability

HyMD is free and open-source, published under a permissive GNU Lesser General Public License v3.0 (LGPLv3). The source code is available at [/github.com/Cascella-Group-UiO/HyMD](https://github.com/Cascella-Group-UiO/HyMD). Documentation, usage guides, and tutorials can be accessed via [cascella-group-uio.github.io/HyMD](https://cascella-group-uio.github.io/HyMD).

# Acknowledgements

This work was supported by the Research Council of Norway through the Centre of Excellence *Hylleraas Centre for Quantum Molecular Sciences*  (grant number 262695), by the Norwegian Supercomputing Program (NOTUR) (grant number NN4654K), and by the Deutsche Forschungsgemeinschaft (DFG) within the project B5 of the TRR-146 (project number 233630050).

# References
