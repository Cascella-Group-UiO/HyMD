import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np
import scipy.cluster.hierarchy as hcl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import argparse
from tqdm import tqdm
import warnings
import dask
import dask.multiprocessing

dask.config.set(scheduler="processes")
warnings.filterwarnings("ignore")


def explore_methods(
    h5mdfile, grofile, cutoff, selection, solvent_name, skip, traj_in_memory
):
    u = mda.Universe(grofile, h5mdfile, in_memory=traj_in_memory)
    ts = u.trajectory[skip]

    # select atoms that are not the solvent
    not_w = u.select_atoms("not name " + solvent_name)

    # get CM for each mol (assumes each ResID is a mol)
    mol_cms = []
    for mol in not_w.split("residue"):
        mol_cms.append(mol.select_atoms(selection).centroid())

    # compute distance matrix
    n = len(mol_cms)
    cond_distmat = np.zeros((int((n * n - n) / 2),), dtype=np.float64)
    distances.self_distance_array(
        np.array(mol_cms), box=ts.dimensions, result=cond_distmat, backend="OpenMP"
    )

    if not os.path.exists("./agg_tests"):
        os.mkdir("./agg_tests")

    for method in tqdm(
        [
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        ]
    ):
        Z = hcl.linkage(cond_distmat, method)
        plt.clf()
        hcl.dendrogram(
            Z,
            leaf_rotation=90.0,  # rotates the x axis labels
            leaf_font_size=6.0,  # font size for the x axis labels
            show_leaf_counts=False,
            no_labels=True,
        )
        plt.axhline(cutoff, linestyle="--")
        plt.savefig(f"./agg_tests/{method}.pdf", bbox_inches="tight")

        # build the clusters and print them to file
        clusters = hcl.fcluster(Z, cutoff, criterion="distance")
        not_w.residues.resids = clusters

        not_w.write(f"./agg_tests/{method}.pdb")


def compute_clusters(
    print_sel, at_sel, ts, cutoff, linkage_method, save_snaps, plot_dendrograms
):
    # get CM for each mol (assumes each ResID is a mol)
    mol_cms = []
    for sel in at_sel:
        mol_cms.append(sel.centroid())

    # compute distance matrix
    n = len(mol_cms)
    cond_distmat = np.zeros((int((n * n - n) / 2),), dtype=np.float64)
    distances.self_distance_array(
        np.array(mol_cms), box=ts.dimensions, result=cond_distmat, backend="OpenMP"
    )

    # get dendrogram
    Z = hcl.linkage(cond_distmat, linkage_method)

    if plot_dendrograms:
        plt.clf()
        hcl.dendrogram(
            Z,
            leaf_rotation=90.0,  # rotates the x axis labels
            leaf_font_size=6.0,  # font size for the x axis labels
            show_leaf_counts=False,
            no_labels=True,
        )
        plt.axhline(cutoff, linestyle="--")
        plt.savefig(f"./dendrograms/snap_{ts.frame}.pdf", bbox_inches="tight")

    # build the clusters
    clusters = hcl.fcluster(Z, cutoff, criterion="distance")

    if save_snaps:
        if len(print_sel.residues.resids) > len(clusters):
            resids = np.full((len(print_sel.residues.resids)), np.max(clusters) + 1)
            resids[:len(clusters)] = clusters
        elif len(print_sel.residues.resids) == len(clusters):
            resids = clusters
        else:
            raise AssertionError("Something is wrong with your selection")
        print_sel.residues.resids = resids

        print_sel.write(f"./colored_pdbs/snap_{ts.frame}.pdb")

    return clusters


def aggregates_clustering(
    h5mdfile,
    grofile,
    cutoff,
    selection,
    skip,
    stride,
    end,
    solvent_name,
    linkage_method,
    save_snaps,
    plot_dendrograms,
    traj_in_memory,
    save_solvent,
):
    u = mda.Universe(grofile, h5mdfile, in_memory=traj_in_memory)

    # select atoms that are not the solvent
    not_w = u.select_atoms("not name " + solvent_name)

    if save_solvent:
        print_sel = u.select_atoms("all")
    else:
        print_sel = not_w

    # get CM for each mol (assumes each ResID is a mol)
    at_sel = []
    for mol in not_w.split("residue"):
        at_sel.append(mol.select_atoms(selection))

    # create directories
    if not os.path.exists("./dendrograms") and plot_dendrograms:
        os.mkdir("./dendrograms")

    if not os.path.exists("./colored_pdbs") and save_snaps:
        os.mkdir("./colored_pdbs")

    # compute clusters for each snapshot
    job_list = []
    frames = []
    for ts in tqdm(u.trajectory[skip:end:stride]):
        frames.append(ts.frame)
        job_list.append(
            dask.delayed(
                compute_clusters(
                    print_sel,
                    at_sel,
                    ts,
                    cutoff,
                    linkage_method,
                    save_snaps,
                    plot_dendrograms,
                )
            )
        )

    clusters = dask.compute(job_list)

    n_clusters = []
    clust_sizes = []
    all_sizes = []
    for c in clusters[0]:
        # get the number of clusters and sizes
        unique_clusts, clust_counts = np.unique(c, return_counts=True)
        n_clusters.append(len(unique_clusts))

        clust_sizes.append(clust_counts)
        all_sizes += clust_counts.tolist()

    # based on cluster sizes get occurence of each size
    sizes, freq = np.unique(all_sizes, return_counts=True)
    freq = freq / len(u.trajectory[skip:end:stride])

    # plot results
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(frames, n_clusters)
    ax1.set_ylabel("Number of aggregates")
    ax1.set_xlabel("Frame")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    xticklabels = [f"{sizes[i]}" for i in range(len(sizes))]
    ax2.bar(xticklabels, freq, width=0.8)
    ax2.set_ylabel("Frequency")
    ax2.set_xlabel("Aggregate size")
    ax2.tick_params("x", labelrotation=60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    description = (
        "Perform aggregate analysis in a .h5md file based on a topology from a .gro"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5md_file", type=str, help="input .h5md file name")
    parser.add_argument("gro_file", type=str, help="input .gro file name")
    parser.add_argument("cutoff", type=float, help="cutoff for the clustering")
    parser.add_argument(
        "--selection",
        type=str,
        default="all",
        dest="selection",
        metavar="selection",
        help="selection used to compute the centroid (default = all) use quotes if needed",
    )
    parser.add_argument(
        "--solvent-name",
        type=str,
        default="W",
        dest="solvent_name",
        metavar="solvent name",
        help="solvent name (default = W, but you should include all ions here as well) use quotes if needed",
    )
    parser.add_argument(
        "--linkage-method",
        type=str,
        default="single",
        help="scipy linkage algorithm (default = single) run with --explore-methods to check all algorithms",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="number of initial frames to skip (default = 0)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="stride when analyzing the trajectory (default = 1)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="final frame to be processed (default = -1)",
    )
    parser.add_argument(
        "--explore-methods",
        action="store_true",
        default=False,
        help="run an exploratory linkage with all methods and print clustered .pdb of frame in subdir agg_tests (can be combined with --skip to select reference frame)",
    )
    parser.add_argument(
        "--save-colored-snap",
        action="store_true",
        default=False,
        help="save snapshots with ResID = cluster number to be colored by ResID in VMD (saved in ./colored_pdbs)",
    )
    parser.add_argument(
        "--save-solvent",
        action="store_true",
        default=False,
        help="when using --save-colored-snap, also save solvent beads in the .pdb",
    )
    parser.add_argument(
        "--plot-dendrograms",
        action="store_true",
        default=False,
        help="plot the dendrograms (saved in ./dendrograms) (use with stride because its ~10x slower)",
    )
    parser.add_argument(
        "--traj-in-memory",
        action="store_true",
        default=False,
        help="load the whole trajectory in memory with MDAanalysis",
    )

    args = parser.parse_args()

    if os.path.splitext(args.h5md_file)[1] != ".h5md":
        raise AssertionError(
            "Trajectory extension should be .h5md. If you are using .H5 please rename it."
        )

    if args.linkage_method not in [
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
        "ward",
    ]:
        raise AssertionError(
            f"Linkage method {args.linkage_method} is not a valid scipy linkage method."
        )

    if args.explore_methods:
        explore_methods(
            args.h5md_file,
            args.gro_file,
            args.cutoff,
            args.selection,
            args.solvent_name,
            args.skip,
            args.traj_in_memory,
        )
    else:
        aggregates_clustering(
            args.h5md_file,
            args.gro_file,
            args.cutoff,
            args.selection,
            args.skip,
            args.stride,
            args.end,
            args.solvent_name,
            args.linkage_method,
            args.save_colored_snap,
            args.plot_dendrograms,
            args.traj_in_memory,
            args.save_solvent
        )
