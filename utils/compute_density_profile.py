import numpy as np
import matplotlib.pyplot as plt
import pathlib
import h5py
from collections.abc import Iterable
import argparse


def h5md_density_profile(file_path, time_steps=(0, -1), species="all",
                         dimension="z", bins=50):
    try:
        in_file = h5py.File(file_path, 'r')
        time = in_file["particles"]["all"]["position"]["time"][...]
        positions = in_file["particles"]["all"]["position"]["value"][...]
        species_index = in_file["parameters"]["vmd_structure"]["indexOfSpecies"][...]
        species_names = in_file["parameters"]["vmd_structure"]["name"][...]
        types = in_file["particles"]["all"]["species"][...]
        in_file.close()
    except IsADirectoryError:
        raise
    except FileNotFoundError:
        raise
    except OSError:
        raise
    
    if species == "all":
        species = tuple([s.decode("UTF-8") for s in species_names])
    else:
        def check_species_present(s, species_list):
            species_list_ = [s_.decode("UTF-8") for s_ in species_list]
            if s not in species_list_:
                raise ValueError(
                    f"No {species} type particles found in the system. "
                    f"Only species present are: {species_list_}."
                )
            
        if isinstance(species, Iterable):
            for s in species:
                check_species_present(s, species_names)
    
    n_steps = time.size
    time_inds = np.arange(n_steps)
    if isinstance(time_steps, int):
        time_steps = (time_steps, time_steps)
    try:
        time_steps_ = []
        for t in time_steps:
            time_steps_.append(time_inds[t])
        time_steps = tuple(time_steps_)
    except (TypeError, IndexError) as e:
        raise TypeError(
            f"Could not interpret time_steps {time_steps} as a list of "
            f"integers."
        ) from e
    

    positions_slice = positions[time_steps[0]:time_steps[1], ...]
    n_steps_average = positions_slice.shape[0]
    dim = 0 if dimension == "x" else (1 if dimension == "y" else 2)
    max_pos = np.max(positions_slice[..., dim])
    min_pos = np.min(positions_slice[..., dim])
    
    for s in species:
        t = species_index[np.where(species_names == np.string_(s))][0]
        ind_t = np.where(types == t)
        hist, bin_edges = np.histogram(
            positions_slice[:, ind_t, dim], bins=bins, 
            range=(min_pos, max_pos),
        )
        hist = hist.astype(np.float64) / float(n_steps_average)
        edge_width = np.mean(np.diff(bin_edges))
        bin_edges_ = bin_edges[:-1] + 0.5 * edge_width
        ax.plot(bin_edges_, hist, label=str(s)+' '+str(time_steps))
    
    ax.set_xlabel(f"{dimension} position / nm", fontsize=13)
    ax.set_ylabel("average number density / 1/nm³", fontsize=13)
    plt.legend()

description = """Plot density profiles from H5MD files

Accumulates the densities from step `time_steps[0]` to `time_steps[1]` and 
averages the resulting density profiles. The `species` argument may specify
one or more species types to calculate the profile for. By default, all 
particle types found in the system are used (corresponds to the default
argument 'all'). The profile is computed across the `dimension` axis.

Parameters
----------
file_path : str or pathlib.Path
    File path to H5MD file.
time_steps : int or tuple of int, optional
    Start and end points for averaging the density. If one number is provided,
    a single frame is considered for the profile and no averaging is done.
species : str or tuple of str, optional
    String or list of strings of particle species to plot the density for.
    By default, all particle types found are treated.
dimension : {'x', 'y', 'z'}
    Dimension to compute the density profile across.
bins : int, optional
    Number of bins used in the density profile histogram.
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument('traj_file', type=str, help='trajectory file with path. Eg: home/sim.h5')
parser.add_argument('--start_frames', type=int, nargs="+",
        help='first frames to be considered. Eg: -50 means range starts from 50th fram from last')
parser.add_argument('--end_frames', type=int, nargs="+",
        help='last frames to be considered. Eg: -1 means range ends at the last frame')
parser.add_argument('--dimension', type=str, default='z', help='axis along which density profile is computed'\
        ' Eg: x [/y/z]')
parser.add_argument('--species', type=str, default='all', help='species whose density profile is computed'\
        ' Eg: all [/W/N]')
parser.add_argument('--bins', type=int, default='60', help='number of bins along axis of plot')
args = parser.parse_args()
home_directory = pathlib.Path.home()
file_path = args.traj_file

fig, ax = plt.subplots(1, 1)
fig.set_figwidth(8)
fig.set_figheight(5)
for i in range(len(args.start_frames)):
    _ = h5md_density_profile(
            file_path, species=args.species, time_steps=(args.start_frames[i], args.end_frames[i]), dimension=args.dimension, bins=args.bins,
    )

plt.show()
