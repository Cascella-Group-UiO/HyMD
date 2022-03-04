import os
import pickle
import numpy as np
import tables
import MDAnalysis as mda
import matplotlib.pyplot as plt
import matplotlib.colors
import itertools
import re
import copy
import argparse
import warnings
from sklearn import metrics as sklm


def find_species_indices(names, species_name):
    """Calculate the index of each particle of species `species_name` in a
    topology HDF5 file or a MDAnalysis trajectory

    If `names` is a numpy.ndarray, searches the `names` array for names
    matching `species_name` and gathers the matching indices. If the input is a
    MDAnalysis universe, the mda atom selection is used.

    Parameters
    ----------
    names : (N,) numpy.ndarray or MDAnalysis.Universe
        Numpy array of particle names of N particles or MDAnalysis Universe
        representing the trajectory.
    species_name : str
        Name of the species type.

    Returns
    -------
    indices : numpy.ndarray
        Indices of particles with names matching `species_name`.
    """
    if isinstance(names, np.ndarray):
        indices = np.where(names == np.string_(species_name))
    elif isinstance(names, mda.Universe):
        indices = names.atoms.select_atoms(f"type {species_name}").indices
    return indices


def parse_molecule_species(names, molecules=None):
    """Sorts particle species into index sets

    Infers sub-species (topology-specific name, differentiating e.g. C1A and
    C2A particles from just type 'C' particles) information about particles in
    the topology file from molecule size, species type, and bond topology.

    Parameters
    ----------
    names : (N,) numpy.ndarray or MDAnalysis.Universe
        Numpy array of particle names of N particles or MDAnalysis Universe
        representing the trajectory.
    molecules : (N,) numpy.ndarray, optional
        Numpy array of molecule numbers for N particles. Only used if `names`
        is a numpy.ndarray type.

    Raises
    ------
    TypeError
        If the input `names` is not type numpy.ndarray or MDAnalysis.Universe.

    Returns
    -------
    species : dict[str, numpy.ndarray]
        Names and indices of particle species.
    """
    if isinstance(names, np.ndarray):
        unique_molecules = np.unique(molecules)
    elif isinstance(names, mda.Universe):
        pass
    else:
        raise TypeError(
            "names must be either type numpy.ndarray or MDAnalysis.Universe."
        )

    dppc = (
        [
            np.string_("type:" + s) for s in ("N", "P", "G", "G",  # head
                                              "C", "C", "C", "C",  # tail 1
                                              "C", "C", "C", "C",  # tail 2
                                              "",  "",)
        ],
        [
            "name:" + s for s in (
                "NC3", "PO4", "GL1", "GL2",  # head
                "C1A", "C2A", "C3A", "C4A",  # tail 1
                "C1B", "C2B", "C3B", "C4B",  # tail 2
            )
        ]
    )
    dmpc = (  # Called DLPC in Martini
        [
            np.string_("type:" + s) for s in ("N", "P", "G", "G",  # head
                                              "C", "C", "C",  # tail 1
                                              "C", "C", "C",  # tail 2
                                              "",  "",  "", "",)
        ],
        [
            "name:" + s for s in (
                "NC3", "PO4", "GL1", "GL2",  # head
                "C1A", "C2A", "C3A",  # tail 1
                "C1B", "C2B", "C3B",  # tail 2
            )
        ]
    )
    dspc = (  # Called DBPC in Martini
        [
            np.string_("type:" + s) for s in ("N", "P", "G", "G",  # head
                                              "C", "C", "C", "C", "C",  # tail1
                                              "C", "C", "C", "C", "C")  # tail2
        ],
        [
            "name:" + s for s in (
                "NC3", "PO4", "GL1", "GL2",  # head
                "C1A", "C2A", "C3A", "C4A", "C5A",  # tail 1
                "C1B", "C2B", "C3B", "C4B", "C5B",  # tail 2
            )
        ]
    )
    dopc = (
        [
            np.string_("type:" + s) for s in ("N", "P", "G", "G",  # head
                                              "C", "C", "D", "C", "C",  # tail1
                                              "C", "C", "D", "C", "C")  # tail2
        ],
        [
            "name:" + s for s in (
                "NC3", "PO4", "GL1", "GL2",  # head
                "C1A", "D2A", "C3A", "C4A",  # tail 1
                "C1B", "D2B", "C3B", "C4B",  # tail 2
            )
        ]
    )
    popg = (
        [
            np.string_("type:" + s) for s in ("L", "P", "G", "G",  # head
                                              "C", "D", "C", "C",  # tail 1
                                              "C", "C", "C", "C",  # tail 2
                                              "",  "",)
        ],
        [
            "name:" + s for s in (
                "GL0", "PO4", "GL1", "GL2",  # head
                "C1A", "D2A", "C3A", "C4A",  # tail 1
                "C1B", "C2B", "C3B", "C4B",  # tail 2
            )
        ]
    )
    toy_lipid = (
        [
            np.string_("type:" + s) for s in ("N", "C", "C", "", "", "", "",
                                              "", "", "", "", "", "", "",)
        ],
        [
            "name:" + s for s in (
                "GL0", "C1A", "C2A",
            )
        ]
    )
    w = (
        [
            np.string_("type:" + s) for s in ("W", "", "", "", "", "", "",
                                              "",  "", "", "", "", "", "",)
        ],
        [
            "name:" + s for s in (
                "W",
            )
        ]
    )

    species = {}
    for d in (dppc, dmpc, dspc, dopc, popg, toy_lipid, w):
        t = {s.decode("UTF-8"): None for s in d[0] if s != b"type:"}
        n = {s: None for s in d[1]}
        species = {**species, **{**t, **n}}
    species = {**species, **{
        "lipid:DPPC": None,
        "lipid:DMPC": None,
        "lipid:DSPC": None,
        "lipid:DOPC": None,
        "lipid:POPG": None,
        "lipid:toy": None,
        "solvent": None,
        "all": None,
    }}

    def add_(dictionary, keys, element):
        for key in keys:
            if dictionary[key] is None:
                dictionary[key] = []
            dictionary[key].append(element)

    if isinstance(names, np.ndarray):
        species["all"] = np.arange(names.size)
        for m in unique_molecules:
            mol = np.where(molecules == m)
            n_mol = mol[0].size
            n = [
                bytes("type:" + s.decode("UTF-8"), "UTF-8") for s in names[mol]
            ]
            if np.array_equal(n, dmpc[0][:n_mol]) and n_mol == 10:
                add_(species, ("name:NC3", "type:N", "lipid:DMPC"), mol[0][0])
                add_(species, ("name:PO4", "type:P", "lipid:DMPC"), mol[0][1])
                add_(species, ("name:GL1", "type:G", "lipid:DMPC"), mol[0][2])
                add_(species, ("name:GL2", "type:G", "lipid:DMPC"), mol[0][3])
                add_(species, ("name:C1A", "type:C", "lipid:DMPC"), mol[0][4])
                add_(species, ("name:C2A", "type:C", "lipid:DMPC"), mol[0][5])
                add_(species, ("name:C3A", "type:C", "lipid:DMPC"), mol[0][6])
                add_(species, ("name:C1B", "type:C", "lipid:DMPC"), mol[0][7])
                add_(species, ("name:C2B", "type:C", "lipid:DMPC"), mol[0][8])
                add_(species, ("name:C3B", "type:C", "lipid:DMPC"), mol[0][9])
            elif np.array_equal(n, dppc[0][:n_mol]) and n_mol == 12:
                add_(species, ("name:NC3", "type:N", "lipid:DPPC"), mol[0][0])
                add_(species, ("name:PO4", "type:P", "lipid:DPPC"), mol[0][1])
                add_(species, ("name:GL1", "type:G", "lipid:DPPC"), mol[0][2])
                add_(species, ("name:GL2", "type:G", "lipid:DPPC"), mol[0][3])
                add_(species, ("name:C1A", "type:C", "lipid:DPPC"), mol[0][4])
                add_(species, ("name:C2A", "type:C", "lipid:DPPC"), mol[0][5])
                add_(species, ("name:C3A", "type:C", "lipid:DPPC"), mol[0][6])
                add_(species, ("name:C4A", "type:C", "lipid:DPPC"), mol[0][7])
                add_(species, ("name:C1B", "type:C", "lipid:DPPC"), mol[0][8])
                add_(species, ("name:C2B", "type:C", "lipid:DPPC"), mol[0][9])
                add_(species, ("name:C3B", "type:C", "lipid:DPPC"), mol[0][10])
                add_(species, ("name:C4B", "type:C", "lipid:DPPC"), mol[0][11])
            elif np.array_equal(n, dspc[0][:n_mol]) and n_mol == 14:
                add_(species, ("name:NC3", "type:N", "lipid:DSPC"), mol[0][0])
                add_(species, ("name:PO4", "type:P", "lipid:DSPC"), mol[0][1])
                add_(species, ("name:GL1", "type:G", "lipid:DSPC"), mol[0][2])
                add_(species, ("name:GL2", "type:G", "lipid:DSPC"), mol[0][3])
                add_(species, ("name:C1A", "type:C", "lipid:DSPC"), mol[0][4])
                add_(species, ("name:C2A", "type:C", "lipid:DSPC"), mol[0][5])
                add_(species, ("name:C3A", "type:C", "lipid:DSPC"), mol[0][6])
                add_(species, ("name:C4A", "type:C", "lipid:DSPC"), mol[0][7])
                add_(species, ("name:C5A", "type:C", "lipid:DSPC"), mol[0][8])
                add_(species, ("name:C1B", "type:C", "lipid:DSPC"), mol[0][9])
                add_(species, ("name:C2B", "type:C", "lipid:DSPC"), mol[0][10])
                add_(species, ("name:C3B", "type:C", "lipid:DSPC"), mol[0][11])
                add_(species, ("name:C4B", "type:C", "lipid:DSPC"), mol[0][12])
                add_(species, ("name:C5B", "type:C", "lipid:DSPC"), mol[0][13])
            elif np.array_equal(n, dopc[0][:n_mol]) and n_mol == 12:
                add_(species, ("name:NC3", "type:N", "lipid:DOPC"), mol[0][0])
                add_(species, ("name:PO4", "type:P", "lipid:DOPC"), mol[0][1])
                add_(species, ("name:GL1", "type:G", "lipid:DOPC"), mol[0][2])
                add_(species, ("name:GL2", "type:G", "lipid:DOPC"), mol[0][3])
                add_(species, ("name:C1A", "type:C", "lipid:DOPC"), mol[0][4])
                add_(species, ("name:D2A", "type:D", "lipid:DOPC"), mol[0][5])
                add_(species, ("name:C3A", "type:C", "lipid:DOPC"), mol[0][6])
                add_(species, ("name:C4A", "type:C", "lipid:DOPC"), mol[0][7])
                add_(species, ("name:C1B", "type:C", "lipid:DOPC"), mol[0][8])
                add_(species, ("name:D2B", "type:D", "lipid:DOPC"), mol[0][9])
                add_(species, ("name:C3B", "type:C", "lipid:DOPC"), mol[0][10])
                add_(species, ("name:C4B", "type:C", "lipid:DOPC"), mol[0][11])
            elif np.array_equal(n, popg[0][:n_mol]) and n_mol == 10:
                add_(species, ("name:GL0", "type:L", "lipid:POPG"), mol[0][0])
                add_(species, ("name:PO4", "type:P", "lipid:POPG"), mol[0][1])
                add_(species, ("name:GL1", "type:G", "lipid:POPG"), mol[0][2])
                add_(species, ("name:GL2", "type:G", "lipid:POPG"), mol[0][3])
                add_(species, ("name:C1A", "type:C", "lipid:POPG"), mol[0][4])
                add_(species, ("name:D2A", "type:D", "lipid:POPG"), mol[0][5])
                add_(species, ("name:C3A", "type:C", "lipid:POPG"), mol[0][6])
                add_(species, ("name:C1B", "type:C", "lipid:POPG"), mol[0][7])
                add_(species, ("name:C2B", "type:C", "lipid:POPG"), mol[0][8])
                add_(species, ("name:C3B", "type:C", "lipid:POPG"), mol[0][9])
            elif np.array_equal(n, toy_lipid[0][:n_mol]) and n_mol == 3:
                add_(species, ("name:GL0", "type:N", "lipid:toy"), mol[0][0])
                add_(species, ("name:C1A", "type:C", "lipid:toy"), mol[0][1])
                add_(species, ("name:C2A", "type:C", "lipid:toy"), mol[0][2])
            elif np.array_equal(n, w[0][:n_mol]) and n_mol == 1:
                add_(species, ("name:W", "type:W", "solvent"), mol[0][0])
            else:
                raise NotImplementedError()

    elif isinstance(names, mda.Universe):
        species_ = copy.deepcopy(species)
        for s in species_:
            s_ = s.replace(":", " ").replace("lipid", "resname")
            if s in ("all", "solvent"):
                if s == "all":
                    species_["all"] = names.atoms.indices
                elif s == "solvent":
                    species_["solvent"] = (
                        names.atoms.select_atoms("type W").indices
                    )
            else:
                species_[s] = (
                    names.atoms.select_atoms(f"{s_}").indices
                )
        species = copy.deepcopy(species_)
    species = {
        k: np.array(v, dtype=int) for k, v in species.items() if v is not None
    }
    return species


def calculate_center_of_mass(positions, simulation_box, indices, axis):
    """Calculate the center of mass of the particles in the `positions` array
    with indices given by the array `indices` along direction `axis` for each
    time step snapshot.

    In order to calculate the position of the center of mass under periodic
    boundary conditions, the dimension in question of the simulation box is
    mapped onto a circle of circumference equal to the box length. Then the
    angles corresponding to particle positions are averaged (not weighted by
    the masses, which are assumed to be equal for all particles), and the
    *center of mass angle* is back-mapped to the corresponding position in the
    simulation box.


    Parameters
    ----------
    positions : (M, N, D) numpy.ndarray or MDAnalysis.Universe
        Array of M timestep snapshots of positions of N particles in D
        dimensions or MDAnalysis Universe representing the trajectory.
    simulation_box : (M, D, D) numpy.ndarray
        Array of M simulation box lengths for each timestep in the form of
        (D, D) numpy.ndarrays.
    indices : (N,) numpy.ndarray
        Array of indices denoting the particles for which to calculate the
        center of mass.
    axis : int
        The direction in which to calculate the center of mass.

    Raises
    ------
    TypeError
        If the input `positions` is not type numpy.ndarray or
        MDAnalysis.Universe.

    Returns
    -------
    center_of_mass : (M,) numpy.ndarray
        The position of the center of mass along the `axis` dimension of the
        simulation box for each time step.
    """
    def compute_center_of_mass_single(p, n, b, c, t):
        p_mapped = 2 * np.pi * p / b
        cos_p_mapped = np.cos(p_mapped)
        sin_p_mapped = np.sin(p_mapped)

        cos_average = np.sum(cos_p_mapped) / n
        sin_average = np.sum(sin_p_mapped) / n

        theta = np.arctan2(-sin_average, -cos_average) + np.pi
        c[t] = b * theta / (2 * np.pi)
        return c

    if isinstance(positions, np.ndarray):
        M = positions.shape[0]
        N = positions.shape[1]
        center_of_mass = np.empty(shape=(M,), dtype=np.float64)
        for time_step in range(M):
            p = positions[time_step, indices, axis]
            center_of_mass = compute_center_of_mass_single(
                p, N, simulation_box[time_step, axis, axis], center_of_mass,
                time_step,
            )

    elif isinstance(positions, mda.Universe):
        M = len(positions.trajectory)
        N = len(positions.atoms)
        center_of_mass = np.empty(shape=(M,), dtype=np.float64)
        for time_step in positions.trajectory:
            p = positions.atoms.positions[indices, axis]
            center_of_mass = compute_center_of_mass_single(
                p, N, time_step.triclinic_dimensions[axis, axis],
                center_of_mass, time_step.frame
            )
    else:
        raise TypeError(
            "positions must be either type numpy.ndarray or "
            "MDAnalysis.Universe."
        )
    return center_of_mass


def load_hymd_simulation(topology_file_path, trajectory_file_path):
    """Load a HyMD trajectory from HyMD H5MD files into memory

    Parameters
    ----------
    topology_file_path : str
        HyMD HDF5 input file path.
    trajectory_file_path : str
        HyMD H5MD output trajectory file path.

    Returns
    -------
    topology_file_hdf5 : tables.file
        In-memory representation of HDF5 file `topology_file_path`.
    trajectory_file_hdf5 : tables.file
        In-memory representation of HDF5 file `trajectory_file_path`.
    """
    topology_file_hdf5 = tables.open_file(
        topology_file_path, driver="H5FD_CORE",
    )
    trajectory_file_hdf5 = tables.open_file(
        trajectory_file_path, driver="H5FD_CORE",
    )
    return topology_file_hdf5, trajectory_file_hdf5


def load_gromacs_simulation(topology_file_path, trajectory_file_path):
    """Load a Gromacs trajectory from .gro and .trr files using MDAnalysis

    Parameters
    ----------
    topology_file_path : str
        Gromacs .gro file path.
    trajectory_file_path : str
        Gromacs .trr trajectory file path.

    Returns
    -------
    universe : MDAnalysis.Universe
        MDAnalysis Universe representing the Gromacs simulation trajectory.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=Warning,
            module="MDAnalysis",
            message=r"Failed to guess the mass for the following atom types",
        )
        universe = mda.Universe(topology_file_path, trajectory_file_path)
    return universe


def compute_centered_histogram(
    centers, positions, simulation_box, species, axis, bins=10, density=False,
    symmetrize=False, skip_first=0, skip=1, frames=None, time=None,
    file_names="<unknown file>", silent=False, range_=None,
):
    """Calculate the density histogram along a direction of positions at
    specified indices realative to a given simlulation box center

    Calculates the average histogram of `positions` along direction `axis`
    centered at position `center` in the `simulation_box`. Only positions
    satisfying the condition specified in `indices` are counted.

    Parameters
    ----------
    centers : (M,) numpy.ndarray
        Array of centers of the simulation box relative to which the histogram
        is calculated for each time step.
    positions : (M, N, D) numpy.ndarray or MDAnalysis.Universe
        Array of M timestep snapshots of positions of N particles in D
        dimensions or MDAnalysis Universe representing the trajectory.
    simulation_box : (M, D, D) numpy.ndarray
        Array of M timestep snapshots of simulation box lengths, in the format
        of (D, D) numpy.ndarrays with the diagonal containing the dimension
        box lengths.
    species : dict[str, numpy.ndarray]
        Dictionary of indices for each species of particle in the system.
    axis : int
        The direction in which to calculate the center of mass.
    bins : int, optional
        Number of bins used in the histogram.
    density : bool, optional
        If True, the result is the value of the probability density function at
        the bin, normalized such that the integral over the range is 1. If
        False, the result will contain the number of samples in each bin.
    symmetrize : bool, optional
        Symmetrize the histograms by averaging the histogram and the flipped
        histogram.
    skip_first : int, optional
        Skip the first `skip_first` time steps, start calculating histograms at
        frame `skip_first` + 1.
    skip : int, optional
        Consider only every `skip` frames when calculating the average
        histograms.
    frames : int, optional
        Only consider `frames` number of frames, starting at `skip_first` (with
        step size `skip`).
    time : (M,) np.ndarray, optional
        Array of times for each frame in the simulation.
    range_ : float, optional
        X-axis range used to compute the histogram.
    file_names : str, optional
        Name of the files containing the trajectory data for printing to
        terminal.
    silent : bool, optional
        If True, no output is printed.

    Raises
    ------
    TypeError
        If the input `positions` is not type numpy.ndarray or
        MDAnalysis.Universe.

    Returns
    -------
    histogram_bins : (M,) numpy.ndarray
        The bins used in the histogram.
    histograms : dict[str, (M,) numpy.ndarray]
        The value of the histogram at each bin for each species in the
        simulation.
    """
    if range_ is None:
        range_ = np.min(simulation_box[:, axis, axis])
        if isinstance(positions, mda.Universe):
            range_ = range_ / 10.0
        warnings.warn(
            f"No --range specified, using min(simulation_box[:, axis]) by "
            f"default ({range_})"
        )

    histograms = {
        s: np.zeros(shape=(bins,), dtype=np.float64) for s in species
    }

    def compute_histogram_single(p, c, b, s, h, d, bins, box, axis_range):
        p += 0.5 * b - c
        p[p > b] -= b
        p[p < 0.0] += b
        for ss, ind in s.items():
            hist, _ = np.histogram(p[ind], bins=bins, density=d)
            if d:
                scaling_factor = 1.0
            else:
                box[axis, axis] = axis_range
                scaling_factor = (
                    np.prod(np.diag(box)) / len(bins)
                )
            h[ss] += hist / scaling_factor
        return h

    if not silent:
        print(f"  → Computing histograms from {file_names}:")
    if isinstance(positions, np.ndarray):
        H = 0
        L = positions.shape[0]
        M = min(L - 1, skip_first + frames * skip) if frames is not None else L
        for time_step in range(skip_first, M, skip):
            p = positions[time_step, :, axis]

            box_length = simulation_box[time_step, ...]
            box_length_axis = simulation_box[time_step, axis, axis]
            box_mid = 0.5 * box_length_axis
            x_start = box_mid - 0.5 * range_
            x_end = box_mid + 0.5 * range_
            range__ = (x_start, x_end)
            axis_range = x_end - x_start

            histogram_bins = np.histogram_bin_edges(
                np.zeros(shape=(1,)), bins=bins, range=range__,
            )
            histograms = compute_histogram_single(
                p, centers[time_step], box_length_axis, species, histograms,
                density, histogram_bins, box_length, axis_range,
            )
            last_frame = time_step
            H += 1

        if not silent:
            if time[skip_first] > 1000.0:
                t0 = time[skip_first] / 1000.0
                t0_unit = "ns"
            else:
                t0 = time[skip_first]
                t0_unit = "ps"
            if time[last_frame] > 1000.0:
                t1 = time[last_frame] / 1000.0
                t1_unit = "ns"
            else:
                t1 = time[last_frame]
                t1_unit = "ps"

            print(
                f"Calculated histograms for {H} frames (first frame "
                f"{skip_first}, last frame {last_frame}, skipping every "
                f"{skip}) starting at {t0:.3f} {t0_unit}, ending at {t1:.3f} "
                f"{t1_unit}."
            )
    elif isinstance(positions, mda.Universe):
        H = 0
        L = len(positions.trajectory)
        M = min(L - 1, skip_first + frames * skip) if frames is not None else L
        for time_step in range(skip_first, M, skip):
            p = positions.trajectory[time_step].positions[:, axis] / 10.0

            box_length = simulation_box[time_step, ...] / 10.0
            box_length_axis = simulation_box[time_step, axis, axis] / 10.0
            box_mid = 0.5 * box_length_axis
            x_start = box_mid - 0.5 * range_
            x_end = box_mid + 0.5 * range_
            range__ = (x_start, x_end)
            axis_range = x_end - x_start

            histogram_bins = np.histogram_bin_edges(
                np.zeros(shape=(1,)), bins=bins, range=range__,
            )
            histograms = compute_histogram_single(
                p, centers[time_step] / 10.0, box_length_axis, species,
                histograms, density, histogram_bins, box_length, axis_range,
            )
            last_frame = time_step
            H += 1
        if not silent:
            if positions.trajectory[skip_first].time > 1000.0:
                t0 = positions.trajectory[skip_first].time / 1000.0
                t0_unit = "ns"
            else:
                t0 = positions.trajectory[skip_first].time
                t0_unit = "ps"
            if positions.trajectory[last_frame].time > 1000.0:
                t1 = positions.trajectory[last_frame].time / 1000.0
                t1_unit = "ns"
            else:
                t1 = positions.trajectory[last_frame].time
                t1_unit = "ps"
            print(
                f"Calculated histograms for {H} frames (first frame "
                f"{skip_first}, last frame {last_frame}, skipping every "
                f"{skip}) starting at {t0:.3f} {t0_unit}, ending at {t1:.3f} "
                f"{t1_unit}."
            )
    else:
        raise TypeError(
            "positions must be either type numpy.ndarray or "
            "MDAnalysis.Universe."
        )

    for s in histograms:
        histograms[s] /= float(H)

    if symmetrize:
        for s in histograms:
            histograms[s] = 0.5 * (histograms[s] + np.flip(histograms[s]))

    histogram_bins -= 0.5 * np.mean(np.diff(histogram_bins))
    return histogram_bins[1:], histograms


def plot_histogram(
    bin_midpoints, histogram, figwidth=4.0, figheight=3.0, show=True,
    ignore=None, one_side=False, xlim=None, ylim=None, vmd_colors=False,
    remove_xlabel=False, remove_ylabel=False, remove_xticks=False,
    remove_yticks=False, tight=False, no_marker=None, remove_legend=False,
):
    """Visualize a histogram

    Parameters
    ----------
    bin_midpoints : (N,) numpy.ndarray
        Numpy array containing N histogram bin midpoints. Note that this is not
        the same as the histogram_bins array used in numpy.histogram, as that
        one represents the N+1 right edges of the bins, whereas `bin_midpoints`
        are the middle points of each N bins.
    histogram : dict[str, (N,) numpy.ndarray]
        Dictionary of species names and histogram values.
    figwidth : float, optional
        Width of the Matplotlib figure.
    figheight : float, optional
        Height of the Matplotlib figure.
    show : bool, optional
        Show the figure if True.
    ignore : list[str], optional
        Species names (keys in `histogram`) to ignore in plot. Evaluated as
        regular expressions.
    one_side : bool, optional
        Only plot one side of the profile if True.
    xlim : list[float], optional
        x-axis limits used in the plot.
    ylim : list[float], optional
        y-axis limits used in the plot.
    remove_xlabel : bool, optional
        Do not display a label on the x axis.
    remove_ylabel : bool, optional
        Do not display a label on the y axis.
    remove_xticks : bool, optional
        Do not display tick labels on the x axis.
    remove_yticks : bool, optional
        Do not display tick labels on the y axis.
    tight : bool, optional
        Use matplotlib tight_layout for the figure.
    vmd_colors : bool, optional
        If True, plot the bead type profiles using the same color map as used
        by default in VMD for that corresponding bead name.
    no_marker : bool, optional
        If True, do not show markers on lines in the plot.
    remove_legend : bool, optional
        If True, do not show the legend on the plot.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Matplotlib figure used as canvas.
    ax : matplotlib.pyplot.Axes
        Matplotlib axes used in the plot.
    """
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)

    colors = matplotlib.colors.TABLEAU_COLORS
    markers = (
        ".", "o", "v", "^", ">", "<", "s", "p", "x", "D", "H", "1", "2", "3",
    )
    colors_cycle = itertools.cycle(colors)
    markers_cycle = itertools.cycle(markers)

    ignore = [re.compile(i) for i in ignore]
    keys = list(histogram.keys())
    for k in histogram.keys():
        for i in ignore:
            if re.fullmatch(i, k) is not None:
                keys.remove(k)

    N = bin_midpoints.size
    if np.mod(N, 2) == 2:
        bin_midpoints -= np.mean(bin_midpoints[N//2 - 1:N//2 + 1])
    else:
        bin_midpoints -= bin_midpoints[N//2]

    for s in keys:
        label = s
        marker = next(markers_cycle) if len(histogram[s]) < 26 else None

        if vmd_colors and "type" in s:
            if s == "type:N":
                color = (0.0, 0.0, 1.0)  # blue
                label = "N"
                marker = "o"
            elif s == "type:C":
                color = (0.25, 0.75, 0.75)  # cyan
                label = "C"
                marker = "x"
            elif s == "type:P":
                color = (0.5, 0.5, 0.2)  # tan
                label = "P"
                marker = "^"
            elif s == "type:G":
                color = (1.0, 0.6, 0.6)  # pink
                label = "G"
                marker = "v"
            elif s == "type:W":
                color = (0.0, 0.0, 0.0)  # black
                label = "W"
                marker = "s"
            elif s == "type:D":
                color = (0.3, 0.8, 0.8)
                label = "D"
                marker = "<"
            elif s == "type:L":
                color = (0.3, 0.3, 0.3)
                label = "L"
                marker = ">"
        else:
            color = next(colors_cycle)

        if no_marker is not None:
            if no_marker:
                marker = ""

        ax.plot(
            bin_midpoints, histogram[s], color=color, label=label,
            marker=marker,
        )

    fontsize = 15
    plt.subplots_adjust(bottom=0.2, left=0.22, right=0.97, top=0.97)

    if not remove_legend:
        ax.legend(loc="best", fontsize=fontsize, framealpha=0)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    if not remove_xlabel:
        ax.set_xlabel("Position along normal, nm", fontsize=fontsize)
    if not remove_ylabel:
        ax.set_ylabel("Number density, nm⁻³", fontsize=fontsize)

    if remove_xticks:
        # https://stackoverflow.com/a/4762002/4179419
        [t.set_color("white") for t in ax.xaxis.get_ticklabels()]
    if remove_yticks:
        [t.set_color("white") for t in ax.yaxis.get_ticklabels()]

    if one_side:
        x_lower, x_upper = ax.get_xlim()
        ax.set_xlim((0.0, x_upper))

    if xlim is not None:
        ax.set_xlim([float(x) for x in xlim])

    if ylim is not None:
        ax.set_ylim([float(y) for y in ylim])

    if tight:
        fig.set_tight_layout(True)

    if show:
        plt.show()

    return fig, ax


def compute_histogram_fitness(
    bin_midpoints, histogram_reference, histogram_test, ignore=None,
    resolution="names", area_per_lipid_test=None, area_per_lipid_ref=None,
):
    """Computes various metrics of the similarity between two histograms

    Metrics used:
    - "MSE": Mean squared error
    - "RMSE": Root mean squared error
    - "MAE": Mean absolute error
    - "MAPE": Mean absolute percentage error
    - "SMAPE": Symmetric mean absolute percentage error
    - "R2": Coefficient of determinantion

    Parameters
    ----------
    bin_midpoints : (N,) numpy.ndarray
        Numpy array containing N histogram bin midpoints. Note that this is not
        the same as the histogram_bins array used in numpy.histogram, as that
        one represents the N+1 right edges of the bins, whereas `bin_midpoints`
        are the middle points of each N bins.
    histogram_reference : dict[str, (N,) numpy.ndarray]
        Dictionary of species names and histogram values for the referece,
        ground truth.
    histogram_test : dict[str, (N,) numpy.ndarray]
        Dictionary of species names and histogram values for the test data.
    ignore : list[str], optional
        Species names (keys in `histogram_reference` and `histogram_test`) to
        ignore in the calculation of the fitness value. Evaluated as regular
        expressions.
    resolution : {"names", "types"}, optional
        If `resolution` is "names", the specific Martini particle subspecies
        names (e.g. NC3, PO4, C1A, D2B, etc.) are compared in the fitness. If
        `resolution` is "types", particle types (e.g. N, P, G, C, etc.) are
        compared in the fitness.

    Raises
    ------
    ValueError
        If `resolution` is not "names" or "types".

    Returns
    -------
    fitness : dict[str, np.ndarray]
        Fitness calculated for each M species in `histogram_reference` and
        `histogram_test` in addition to the sum total fitness across all
        species for each of the K available metrics.
    """
    metrics = ("MSE", "RMSE", "MAE", "MAPE", "R2", "SMAPE")

    ignore = [re.compile(i) for i in ignore] if ignore is not None else []
    keys = list(({**histogram_reference, **histogram_test}).keys())
    for k in histogram_reference.keys():
        for i in ignore:
            if re.fullmatch(i, k) is not None:
                keys.remove(k)
    for k_test in histogram_test.keys():
        if k_test not in histogram_reference:
            keys.remove(k_test)
    for k_ref in histogram_reference.keys():
        if k_ref not in histogram_test:
            keys.remove(k_ref)

    N = bin_midpoints.size
    accuracy = {s: {m: None for m in metrics} for s in keys}
    if area_per_lipid_ref is not None and area_per_lipid_test is not None:
        accuracy_add = {"area_per_lipid": {m: None for m in metrics}}
        accuracy = {**accuracy, **accuracy_add}
        keys = keys + ["area_per_lipid"]

    for s in keys:
        if s == "area_per_lipid":
            y = np.tile(
                np.array(  # area per lipid in Å
                    [100 * np.mean(area_per_lipid_test)], dtype=np.float64
                ), 2,
            ).T
            print("Area per lipid: ", y[0])
            y_true = np.array([63.0, 63.0], dtype=np.float64)

        else:
            y = histogram_test[s]
            y_true = histogram_reference[s]

        accuracy[s]["MSE"] = sklm.mean_squared_error(
            y_true, y, squared=True
        )
        accuracy[s]["RMSE"] = sklm.mean_squared_error(
            y_true, y, squared=False
        )
        accuracy[s]["MAE"] = sklm.mean_absolute_error(
            y_true, y
        )
        accuracy[s]["MAPE"] = sklm.mean_absolute_percentage_error(
            y_true, y
        )
        accuracy[s]["R2"] = sklm.r2_score(
            y_true, y
        )
        accuracy[s]["SMAPE"] = 100.0 / N * np.sum(
            np.divide(
                np.abs(y - y_true), np.abs(y_true) + np.abs(y),
                out=np.zeros_like(y, dtype=np.float64),
                where=(np.abs(y_true) + np.abs(y)) > np.finfo(np.float64).eps,
            )
        )
        if s == "area_per_lipid":
            accuracy[s]["SMAPE"] = accuracy[s]["MSE"]

    if resolution == "names":
        total_re = re.compile("name:.*")
    elif resolution == "types":
        total_re = re.compile("type:.*")
    else:
        raise ValueError("resolution must be either 'names' or 'types'.")
    for k in accuracy.keys():
        if re.fullmatch(total_re, k) is None:
            if k != "area_per_lipid":
                keys.remove(k)

    accuracy["total"] = {}
    for m in metrics:
        accuracy["total"][m] = np.mean([accuracy[k][m] for k in keys])

    return accuracy


def calculate_area_per_lipid(
    positions, simulation_box, axis, parser, ref, time=None, molecules=None,
    names=None,
):
    """Compute the area per lipid for each simulation frame

    Parameters
    ----------
    positions : (M, N, D) numpy.ndarray or MDAnalysis.Universe
        Array of M timestep snapshots of positions of N particles in D
        dimensions or MDAnalysis Universe representing the trajectory.
    simulation_box : (M, D, D) numpy.ndarray
        Array of M timestep snapshots of simulation box lengths, in the format
        of (D, D) numpy.ndarrays with the diagonal containing the dimension
        box lengths.
    axis : int
        The transverse direction of the bilayer.
    parser : Argparse.Namespace
        Parsed command line arguments given to hymd_optimize.
    time : (M,) numpy.ndarray, optional
        Array of times for each frame in the simulation.
    molecules : (N,) numpy.ndarray, optional
        Array of molecule indices for each of N particles.
    names : (N,) numpy.ndarray, optional
        Array of particle names for each of N particles.
    ref : bool
        True if this is the reference simulation, False if this is the test
        simulation.

    Returns
    -------
    area_per_lipid : (M,) numpy.ndarray
        Area per lipid value for each of M simulation frames.
    """
    if isinstance(positions, np.ndarray):
        box = copy.deepcopy(simulation_box)  # HyMD output is already in nm.
        assert time is not None
        assert molecules is not None
        assert names is not None
        t = time
        inds_lipids = np.where(names != np.string_("W"))
        n_lipids = len(np.unique(molecules[inds_lipids]))
        n_frames = positions.shape[0]
    elif isinstance(positions, mda.Universe):
        box = copy.deepcopy(simulation_box)
        box = box / 10.0  # Convert to nm from Å in MDA.
        t = np.array(
            [ts.time for ts in positions.trajectory], dtype=np.float64,
        )
        lipid_atoms = positions.select_atoms("not name W")
        n_lipids = len(np.unique([a.resid for a in lipid_atoms.atoms]))
        n_frames = len(t)

    M = simulation_box.shape[0]
    area_per_lipid = np.zeros(shape=(M,), dtype=np.float64)
    for i, b in enumerate(box[:, ...]):
        b_ = copy.deepcopy(b)
        b_[axis, axis] = 1.0
        area_per_lipid[i] = np.prod(np.diag(b_))
    area_per_lipid = area_per_lipid / (n_lipids / 2.0)

    """
    fig, axs = plt.subplots(2)
    axs[0].plot(t / 1000.0, area_per_lipid, "o")
    axs[0].set_xlabel("time, ns")
    axs[0].set_ylabel("area per lipid, nm²")
    axs[0].set_ylim(0.57, 0.64)
    axs[1].plot(t / 1000.0, box[:, 0, 0], "rx", label="xy")
    axs[1].plot(t / 1000.0, box[:, 2, 2], "bo", label="z")
    axs[1].set_xlabel("time, ns")
    axs[1].set_ylabel("box side length, nm")
    axs[1].legend()
    plt.show()
    """

    if ref is False:
        skip_ = parser.skip
        skip_first_ = parser.skip_first
        frames_ = parser.frames
    else:
        skip_ = parser.skip_ref
        skip_first_ = parser.skip_first_ref
        frames_ = parser.frames_ref

    M = min(n_frames - 1, skip_first_ + frames_ * skip_) if frames_ is not None else n_frames  # noqa: E501
    return area_per_lipid[range(skip_first_, M, skip_)]


def action_compute_histogram(parser, ref=False):
    """Compute histogram and area per lipid from H5MD or Gromacs trajectory

    Parameters
    ----------
    parser : Argparse.Namespace
        Parsed command line arguments given to hymd_optimize.
    ref : bool, optional
        Attempt to load the reference topology and reference trajectory files
        if True.

    Returns
    -------
    histogram_bins : (M,) numpy.ndarray
        The bins used in the histogram.
    histograms : dict[str, (M,) numpy.ndarray]
        The value of the histogram at each bin for each species in the
        simulation.
    area_per_lipid : (M,) numpy.ndarray
        Area per lipid for each of M simulation frames.
    """
    if ref:
        axis = parser.axis_ref if parser.axis_ref is not None else parser.axis
        traj_path = parser.ref_traj
        top_path = parser.ref_top
    else:
        axis = parser.axis
        traj_path = parser.traj
        top_path = parser.top

    try:
        top_file_hdf5, traj_file_hdf5 = load_hymd_simulation(
            top_path, traj_path
        )
        names = top_file_hdf5.root.names.read()
        molecules = top_file_hdf5.root.molecules.read()
        positions = traj_file_hdf5.root.particles.all.position.value.read()
        times = traj_file_hdf5.root.particles.all.position.time.read()

        simulation_box_group = traj_file_hdf5.root.particles.all.box
        if isinstance(simulation_box_group.edges, tables.Leaf):
            simulation_box = np.zeros(
                shape=(positions.shape[0], 3, 3,), dtype=np.float64,
            )
            for i in range(positions.shape[0]):
                simulation_box[i, ...] = np.diag(simulation_box_group.edges[:])
        elif isinstance(simulation_box_group.edges, tables.group.Group):
            simulation_box = simulation_box_group.edges.value[...]
        else:
            raise TypeError(
                f"Expected /root/particles/all/box/edges to be either a "
                f"h5py.Group or a h5py.Dataset, not "
                f"{type(simulation_box_group.edges)}"
            )

        carbon_indices = find_species_indices(names, "C")
        carbon_center_of_mass = calculate_center_of_mass(
            positions, simulation_box, carbon_indices, axis,
        )
        species = parse_molecule_species(names, molecules=molecules)

        histogram_bins, histograms = compute_centered_histogram(
            carbon_center_of_mass, positions, simulation_box, species,
            axis=axis, bins=parser.bins, symmetrize=parser.symmetrize,
            density=parser.density, skip=parser.skip,
            skip_first=parser.skip_first, frames=parser.frames,
            time=times, file_names=(
                os.path.abspath(top_path) + ", " + os.path.abspath(traj_path)
            ), range_=parser.range,
        )
        top_file_hdf5.close()
        traj_file_hdf5.close()

        if parser.area_per_lipid:
            area_per_lipid = calculate_area_per_lipid(
                positions, simulation_box, axis, parser, ref=False, time=times,
                molecules=molecules, names=names,
            )
        else:
            area_per_lipid = None

    except tables.exceptions.HDF5ExtError:
        universe = load_gromacs_simulation(top_path, traj_path)
        simulation_box = np.zeros(
            shape=(len(universe.trajectory), 3, 3,), dtype=np.float64,
        )
        for i, ts in enumerate(universe.trajectory):
            simulation_box[i, ...] = ts.triclinic_dimensions
        carbon_indices = find_species_indices(universe, "C")
        carbon_center_of_mass = calculate_center_of_mass(
            universe, simulation_box, carbon_indices, axis
        )
        species = parse_molecule_species(universe, None)
        if ref is False:
            skip_ = parser.skip
            skip_first_ = parser.skip_first
            frames_ = parser.frames
        else:
            skip_ = parser.skip_ref
            skip_first_ = parser.skip_first_ref
            frames_ = parser.frames_ref

        histogram_bins, histograms = compute_centered_histogram(
            carbon_center_of_mass, universe, simulation_box, species,
            axis=axis, bins=parser.bins, symmetrize=parser.symmetrize,
            density=parser.density, skip=skip_, skip_first=skip_first_,
            frames=frames_, range_=parser.range, file_names=(
                (os.path.abspath(top_path) + ", " +
                 os.path.abspath(traj_path))
            ),
        )
        if parser.area_per_lipid:
            area_per_lipid = calculate_area_per_lipid(
                universe, simulation_box, axis, parser, ref=ref,
            )
        else:
            area_per_lipid = None
    return histogram_bins, histograms, area_per_lipid


def action_plot(parser):
    """Execute the 'plot' action of hymd_optimize

    Parameters
    ----------
    parser : Argparse.Namespace
        Parsed command line arguments given to hymd_optimize.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Matplotlib figure used as canvas.
    ax : matplotlib.pyplot.Axes
        Matplotlib axes used in the plot.
    """
    histogram_bins, histograms, area_per_lipid = action_compute_histogram(
        parser
    )
    fig, ax = plot_histogram(
        histogram_bins, histograms, ignore=parser.ignore,
        show=not parser.no_show, one_side=parser.one_side, xlim=parser.xlim,
        ylim=parser.ylim, vmd_colors=parser.vmd_colors,
        remove_xlabel=parser.remove_xlabel, remove_ylabel=parser.remove_ylabel,
        remove_yticks=parser.remove_yticks, remove_xticks=parser.remove_xticks,
        tight=parser.tight, no_marker=parser.no_marker,
        remove_legend=parser.remove_legend,
    )
    """
    import pickle
    pickle.dump(histogram_bins, open("bins_tmp.p", "wb",),)
    pickle.dump(histograms, open("hist_tmp.p", "wb",),)
    """
    return fig, ax


def action_fitness(parser):
    """Execute the 'fitness' action of hymd_optimize

    Parameters
    ----------
    parser : Argparse.Namespace
        Parsed command line arguments given to hymd_optimize.

    Returns
    -------
    fitness : dict[str, np.ndarray]
        Fitness calculated for each M species in `histogram_reference` and
        `histogram_test` in addition to the sum total fitness across all
        species for each of the K available metrics.
    """
    histogram_bins, histograms_test, area_per_lipid_test = (
        action_compute_histogram(parser)
    )
    _, histograms_ref, area_per_lipid_ref = (
        action_compute_histogram(parser, ref=True)
    )
    fitness = compute_histogram_fitness(
        histogram_bins, histograms_ref, histograms_test,
        area_per_lipid_test=area_per_lipid_test,
        area_per_lipid_ref=area_per_lipid_ref, ignore=parser.ignore,
        resolution=parser.resolution,
    )
    return fitness


def action_save_fitness(parser, fitness):
    """Save the computed fitness value(s) to a file

    Parameters
    ----------
    parser : Argparse.Namespace
        Parsed command line arguments given to hymd_optimize.
    fitness :  dict[str, np.ndarray]
        Fitness calculated for the test histograms with respec to the
        reference histograms.
    """
    # if parser.out is not None:
    #     with open(parser.out + ".pickle", "wb") as out_file:
    #         pickle.dump(fitness, out_file)

    metrics = next(iter(fitness.values())).keys()
    M = len(metrics)
    out_str = f"{'=' * 12}{'=' * M * 20}\n"
    out_str += f"{'particle':20}"
    for m in metrics:
        out_str += f"{m:>20}"
    out_str += f"\n{'-' * 20}{'-' * M * 20}\n"

    def sort_key(k):
        for i, s in enumerate(
            ("type", "name", "lipid", "solvent", "all",)
        ):
            if s in k:
                return i
        if "total" in k:
            return 2147483646
        elif "area_per_lipid" in k:
            return 2147483646 - 1
        else:
            return 10

    keys = list(fitness.keys())
    keys.sort(key=sort_key)

    for key in keys:
        metrics = fitness[key]
        out_str += f"{key:20}"
        for fit in metrics.values():
            out_str += f"{fit:20.10g}"
        out_str += "\n"
    out_str += f"{'=' * 20}{'=' * M * 20}"
    if parser.out is not None:
        with open(parser.out, "w") as out_file:
            out_file.write(out_str)
    return out_str


def action_fitness_range(parser, figwidth=4.0, figheight=3.0, show=True):
    """Execute the 'fitness-range' action of hymd_optimize

    Parameters
    ----------
    parser : Argparse.Namespace
        Parsed command line arguments given to hymd_optimize.
    figwidth : float, optional
        Width of the Matplotlib figure.
    figheight : float, optional
        Height of the Matplotlib figure.
    show : bool, optional
        Show the figure if True.

    """
    axis = parser.axis_ref if parser.axis_ref is not None else parser.axis
    traj_path_ref = parser.ref_traj
    top_path_ref = parser.ref_top
    universe = load_gromacs_simulation(top_path_ref, traj_path_ref)
    simulation_box = universe.trajectory[0].dimensions[:3]
    carbon_indices = find_species_indices(universe, "C")
    carbon_center_of_mass = calculate_center_of_mass(
        universe, simulation_box, carbon_indices, axis
    )
    species = parse_molecule_species(universe, None)
    _, histograms_ref = compute_centered_histogram(
        carbon_center_of_mass, universe, simulation_box, species, axis=axis,
        bins=parser.bins, symmetrize=parser.symmetrize,
        density=parser.density, skip=parser.skip_ref,
        skip_first=parser.skip_first_ref, frames=parser.frames_ref,
        file_names=(
            (os.path.abspath(top_path_ref) + ", "
             + os.path.abspath(traj_path_ref))
        ),
    )

    traj_path = parser.traj
    top_path = parser.top
    top_file_hdf5, traj_file_hdf5 = load_hymd_simulation(
        top_path, traj_path
    )
    names = top_file_hdf5.root.names.read()
    molecules = top_file_hdf5.root.molecules.read()
    simulation_box = traj_file_hdf5.root.particles.all.box.edges.read()
    positions = traj_file_hdf5.root.particles.all.position.value.read()
    times = traj_file_hdf5.root.particles.all.position.time.read()
    carbon_indices = find_species_indices(names, "C")

    carbon_center_of_mass = np.zeros(shape=(3, positions.shape[0]))
    for axis in range(3):
        c = calculate_center_of_mass(
            positions, simulation_box, carbon_indices, axis,
        )
        carbon_center_of_mass[axis, :] = c

    species = parse_molecule_species(names, molecules=molecules)

    fitness_range = {0: [], 1: [], 2: [], "time": []}
    for frame in range(parser.frames):
        fitness_range["time"].append(times[frame])
        for axis in range(3):
            histogram_bins, histograms = compute_centered_histogram(
                carbon_center_of_mass[axis, :], positions, simulation_box,
                species, axis=axis, bins=parser.bins,
                symmetrize=parser.symmetrize, density=parser.density,
                skip_first=frame, frames=1, silent=True, time=times,
            )
            fitness = compute_histogram_fitness(
                histogram_bins, histograms_ref, histograms,
                ignore=parser.ignore, resolution=parser.resolution,
            )
            fitness_range[axis].append(fitness)

    if show:
        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(figwidth)
        fig.set_figheight(figheight)

        colors = matplotlib.colors.TABLEAU_COLORS
        markers = (
            ".", "o", "v", "^", ">", "<", "s", "p", "x", "D", "H", "1", "2",
        )
        colors_cycle = itertools.cycle(colors)
        markers_cycle = itertools.cycle(markers)
        for axis in range(3):
            ax.plot(
                times[:parser.frames],
                [f["total"][args.metric] for f in fitness_range[axis]],
                label=["x", "y", "z"][axis], color=next(colors_cycle),
                marker=next(markers_cycle),
            )
        ax.legend(fontsize=15)
        ax.set_xlabel("Time, ps", fontsize=15)
        ax.set_ylabel(f"Fitness ({parser.metric}), dim.less.", fontsize=15)

    top_file_hdf5.close()
    traj_file_hdf5.close()

    if show:
        plt.show()

    return fitness_range


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Tools for HyMD parameter optimization and fitness calculation"
        )
    )
    parser.add_argument(
        "action", type=str, default=None, help="action to perform",
        choices=["plot", "fitness", "fitness-range"],
    )
    parser.add_argument(
        "--out", type=str, default=None, metavar="file name",
        help="output file path",
    )
    parser.add_argument(
        "--no-show", "--noshow", default=False, action="store_true",
        dest="no_show", help="do not show the plot",
    )
    parser.add_argument(
        "--save-plot", action="store_true", default=False, dest="save_plot",
        help="save the histogram plot to this file path",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", default=False, dest="force",
        help="overwrite existing output file path",
    )
    parser.add_argument(
        "--one-side", "--oneside", default=False, action="store_true",
        dest="one_side", help="only plot one side of the membrane",
    )
    parser.add_argument(
        "--xlim", "--x-lim", default=None, dest="xlim", nargs=2,
        help="x-axis limits to use for the plot",
    )
    parser.add_argument(
        "--ylim", "--y-lim", default=None, dest="ylim", nargs=2,
        help="y-axis limits to use for the plot",
    )
    parser.add_argument(
        "--ref-traj", type=str, default=None, dest="ref_traj",
        help="refence trajectory file path (.trr)",
    )
    parser.add_argument(
        "--ref-top", type=str, default=None, dest="ref_top",
        help="refence topology file path (.gro)",
    )
    parser.add_argument(
        "--traj", type=str, default=None,
        help="test trajectory file path (.trr or .H5MD)",
    )
    parser.add_argument(
        "--top", type=str, default=None,
        help="test topology file path (.gro or HyMD-input .H5)",
    )
    parser.add_argument(
        "--bins", type=int, default=25,
        help="number of bins to use in the histograms and histogram plots",
    )
    parser.add_argument(
        "--density", action="store_true",
        help="compute density distribution histograms, not the number density",
    )
    parser.add_argument(
        "--symmetrize", action="store_true", default=False,
        help=(
            "symmetrize the histogram(s) by averaging the histogram(s) and "
            "the flipped histogram."
        ),
    )
    parser.add_argument(
        "--range", type=float, default=None,
        help="histogram x range to consider",
    )
    parser.add_argument(
        "--area-per-lipid", action="store_true", default=False,
        help="Include the area per lipid in the fitness calculation",
        dest="area_per_lipid",
    )
    parser.add_argument(
        "--axis", type=int, choices=[0, 1, 2], default=2,
        help="the direction in which to calculate the histogram(s)",
    )
    parser.add_argument(
        "--axis-ref", type=int, choices=[0, 1, 2], default=None,
        help="the direction in which to calculate the reference histogram(s)",
    )
    parser.add_argument(
        "--ignore", default=[], nargs="+",
        help=(
            "species names to ignore in the calculation of the histograms "
            "(evaluated as list of regex expressions)"
        ),
    )
    parser.add_argument(
        "--resolution", type=str, choices=["types", "names"], default="types",
        help=(
            "if 'names', the specific Martini particle subspecies names (NC3, "
            "PO4, etc.) are compared in the fitness. If 'types', particle "
            "types (N, P, G, etc.) are compared in the fitness"
        ),
    )
    parser.add_argument(
        "--remove-xlabel", action="store_true", default=False,
        dest="remove_xlabel", help="do not show the xlabel in the plot",
    )
    parser.add_argument(
        "--remove-ylabel", action="store_true", default=False,
        dest="remove_ylabel", help="do not show the ylabel in the plot",
    )
    parser.add_argument(
        "--remove-yticks", action="store_true", default=False,
        dest="remove_yticks", help="do not show the y ticks in the plot",
    )
    parser.add_argument(
        "--remove-xticks", action="store_true", default=False,
        dest="remove_xticks", help="do not show the x ticks in the plot",
    )
    parser.add_argument(
        "--remove-legend", action="store_true", default=False,
        dest="remove_legend", help="do not show legend in the plot",
    )
    parser.add_argument(
        "--tight", "-tight", action="store_true", default=False,
        help="use matplotlib tight_layout for the plot",
    )
    parser.add_argument(
        "--no-marker", "-no-marker", "-nomarker", "--nomarker", default=False,
        action="store_true", help="dont show line markers", dest="no_marker",
    )
    parser.add_argument(
        "--skip", type=int, default=1,
        help="consider only every N frames when calculating the histograms",
    )
    parser.add_argument(
        "--skip-first", type=int, default=0,
        help="skip the first N frames when calculating the histograms",
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help=(
            "Only consider N frames, starting at --skip_first (with step "
            "size --skip)."
        ),
    )
    parser.add_argument(
        "--skip-ref", type=int, default=1,
        help=(
            "consider only every N reference frames when calculating the "
            "histograms"
        ),
    )
    parser.add_argument(
        "--skip-first-ref", type=int, default=0,
        help=(
            "skip the first N reference frames when calculating the histograms"
        ),
    )
    parser.add_argument(
        "--frames-ref", type=int, default=None,
        help=(
            "Only consider N reference frames, starting at --skip_first (with "
            "step size --skip)."
        ),
    )
    parser.add_argument(
        "--metric", choices=["MSE", "RMSE", "MAE", "MAPE", "SMAPE", "R2"],
        default="R2", help="which fitness metric to use in the fitness-range",
    )
    parser.add_argument(
        "--vmd-colors", "--vmdcolors", "-vmdcolors", action="store_true",
        default=False, help="use the same colors for beads as default in vmd",
    )

    args = parser.parse_args()

    if args.action == "plot":
        if args.traj is None or args.top is None:
            parser.error(
                "Must specify both --traj and --top with 'plot' action."
            )
        if args.out is None and args.no_show:
            warnings.warn(
                "No --out path specified and --no-show specified, not doing "
                "anything(!)."
            )
        elif args.out is None and not args.no_show:
            warnings.warn(
                "No --out path specified and, not saving plot."
            )
        fig, _ = action_plot(args)
        if args.out is not None:
            if args.force and os.path.exists(args.out):
                print(f"Saving figure to {args.out} (overwriting existing).")
            elif os.path.exists(args.out):
                raise FileExistsError(
                    f"The file {args.out} already exists. To force overwrite, "
                    "use the -f option."
                )
            else:
                print(f"Saving figure to {args.out}.")
            fig.savefig(args.out, format="pdf", transparent=True)

    elif args.action == "fitness":
        if (args.traj is None or args.top is None or
                args.ref_traj is None or args.ref_top is None):
            parser.error(
                "Must specify all of --traj, --top, --ref-traj, and --ref-top "
                "with 'fitness' action."
            )
        if args.out is None:
            warnings.warn(
                "No --out path specified, not saving histograms to file."
            )
        fitness = action_fitness(args)
        fitness_str = action_save_fitness(args, fitness)
        print(fitness_str)

        if args.out is not None:
            if args.force and os.path.exists(args.out):
                print(
                    f"Saving fitness data to {args.out} "
                    f"(overwriting existing)."  # noqa: E501
                )
            elif os.path.exists(args.out):
                raise FileExistsError(
                    f"The file {args.out} already exists. To force overwrite, "
                    "use the -f option."
                )
            else:
                print(f"Saving fitness data to {args.out}.")

    elif args.action == "fitness-range":
        if (args.traj is None or args.top is None or
                args.ref_traj is None or args.ref_top is None):
            parser.error(
                "Must specify all of --traj, --top, --ref-traj, and --ref-top "
                "with 'fitness-range' action."
            )
        if args.out is None:
            warnings.warn(
                "No --out path specified, not saving fitness-range to file."
            )
        fitness_range = action_fitness_range(args, show=not args.no_show)

        if args.out is not None:
            if args.force and os.path.exists(args.out):
                print(
                    f"Saving fitness range to {args.out} "
                    f"(overwriting existing)."
                )
            elif os.path.exists(args.out):
                raise FileExistsError(
                    f"The file {args.out} already exists. To force overwrite, "
                    "use the -f option."
                )
            else:
                print(f"Saving fitness range to {args.out}.")
                with open(args.out, "wb") as out_file:
                    pickle.dump(fitness_range, out_file)
