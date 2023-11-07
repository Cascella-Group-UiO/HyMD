import MDAnalysis as mda
from MDAnalysis.lib import distances
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
import argparse
from tqdm import tqdm
import tomli


def process_topology(topol_path):
    with open(topol_path, "rb") as f:
        topol = tomli.load(f)

    if "include" in topol["system"]:
        for file in topol["system"]["include"]:
            path = os.path.join(os.path.dirname(topol_path), file)
            with open(path, "rb") as f:
                itps = tomli.load(f)
            for mol, itp in itps.items():
                topol[mol] = itp

    return topol


def parse_masses(fname):
    if not fname:
        return None

    masses = {}
    with open(fname, "r") as f:
        for line in f:
            masses[line.split()[0]] = [float(x) for x in line.split()[1:]]
    return masses


def compute_rdfs(
    topol,
    pdbdir,
    agg_size,
    selections,
    name_selections,
    center,
    nbins,
    rmax,
    masses,
    skip_rdfs,
    compute_i_rg,
    compute_pddf,
    save_centered,
    save_agg_only,
):
    # initialize
    rdfs = {}
    for i in range(len(selections)):
        rdfs[i] = np.zeros(nbins)
    if compute_i_rg:
        I1 = []
        I2 = []
        I3 = []
        Rg = []
    if compute_pddf:
        pddf = np.zeros(nbins)

    nsnaps = 0
    for snapshot in tqdm(glob(os.path.join(pdbdir, "*.pdb"))):
        u = mda.Universe(snapshot)

        # based on topology and residues determine size of aggregates
        resids = {}
        for residue in u.residues:
            nummols = int(
                residue.atoms.positions.shape[0] / topol[residue.resname]["atomnum"]
            )
            if residue.resid in resids:
                resids[residue.resid] += nummols
            else:
                resids[residue.resid] = nummols

        # check if we can find an aggregate of the given size
        agg_resids = []
        for k, v in resids.items():
            if v == agg_size:
                agg_resids.append(k)

        # if size was not found, go to next pdb
        if len(agg_resids) == 0:
            continue

        # for each aggregate of given size, accumulate the RDFs
        for resid in agg_resids:
            nsnaps += 1
            box_vectors = u.dimensions

            if not center:
                center_coord = u.select_atoms(f"resid {resid}").positions
            else:
                center_coord = u.select_atoms(f"resid {resid} and " + center).positions

            # get the geometric center using the algorithm:
            # https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
            tetha = (center_coord / box_vectors[:3]) * (2.0 * np.pi)
            xi = np.cos(tetha)
            zeta = np.sin(tetha)

            xi_bar = np.mean(xi, axis=0)
            zeta_bar = np.mean(zeta, axis=0)
            tetha_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi

            cog = (box_vectors[:3] * tetha_bar) / (2.0 * np.pi)

            # since some methods are not PBC aware, we center the
            # aggregate in the box so it does not split in the PBCs
            if save_agg_only or save_centered or compute_i_rg:
                box_center = box_vectors[:3] / 2.0
                u.atoms.translate(box_center - cog)
                u.atoms.wrap(compound="atoms")
                cog = box_center

            if save_agg_only:
                agg_sel = u.select_atoms(f"resid {resid}")

                dirname = f"./agg_only_{agg_size}"

                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                agg_sel.atoms.write(
                    os.path.join(dirname, f"centered_{os.path.basename(snapshot)}")
                )

            if save_centered:
                dirname = f"./centered_{agg_size}"

                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                u.atoms.write(
                    os.path.join(dirname, f"centered_{os.path.basename(snapshot)}")
                )

            # compute the principal moment of inertia and Rg
            if compute_i_rg:
                # set the masses
                agg_sel = u.select_atoms(f"resid {resid}")

                i = 0
                mass_array = []
                while i < len(agg_sel.resnames):
                    name = agg_sel.resnames[i]
                    for bmass in masses[name]:
                        mass_array.append(bmass)
                    i += len(masses[name])

                mass_array = np.array(mass_array)
                agg_sel.masses = mass_array
                total_mass = np.sum(mass_array)

                # get moment of inertia and principal axis
                I = agg_sel.moment_of_inertia()
                UT = agg_sel.principal_axes()  # this is already transposed

                # diagonalizes I
                Lambda = UT.dot(I.dot(UT.T))

                I1.append(Lambda[0][0])
                I2.append(Lambda[1][1])
                I3.append(Lambda[2][2])

                # get radius of gyration
                Rg.append(agg_sel.radius_of_gyration())

            # compute the pair distribution function
            if compute_pddf:
                agg_sel = u.select_atoms(f"resid {resid}")

                n = len(agg_sel)
                cond_distmat = np.zeros((int((n * n - n) / 2),), dtype=np.float64)
                distances.self_distance_array(
                    agg_sel.positions,
                    box=box_vectors,
                    result=cond_distmat,
                    backend="OpenMP",
                )

                pddf_i, pddf_edges = np.histogram(
                    cond_distmat,
                    bins=nbins,
                    range=(0, rmax),
                    density=False,
                )

                pddf += pddf_i

            if not skip_rdfs:
                # create selections for which RDFs will be computed
                for i, sel_string in enumerate(selections):
                    if sel_string.strip().lower() in ["name w", "type w", "name na", "type na", "name cl", "type cl"]:
                        at_sel = u.select_atoms(sel_string)
                    else:
                        at_sel = u.select_atoms(f"resid {resid} and " + sel_string)

                    # Compute the distances between the target atoms and the center
                    distances_matrix = distances.distance_array(
                        cog, at_sel.positions, box=box_vectors, backend="OpenMP"
                    )

                    # Flatten the distances matrix to a 1D array
                    flattened_distances = distances_matrix.flatten()

                    # Compute the RDF using numpy histogram
                    rdf, bin_edges = np.histogram(
                        flattened_distances, bins=nbins, range=(0, rmax), density=False
                    )

                    rdfs[i] += rdf * np.prod(box_vectors[:3]) / len(u.atoms)

    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["font.size"] = 22

    # check if rdfs are skipped
    if not skip_rdfs:
        # Compute the average volume of each shell
        shell_volumes = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

        # Prepare to plot
        mid_bin = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        np.savetxt(f"{agg_size}_bins.txt", mid_bin)

        # Normalize the RDF by dividing by the shell volume and number of snapshots and plot
        for i, sel in enumerate(selections):
            rdfs[i] = rdfs[i] / (shell_volumes * nsnaps)

            plt.plot(mid_bin / 10.0, rdfs[i], linewidth=3, label=name_selections[i])

            np.savetxt(f"{agg_size}_" + sel.replace(" ", "_") + ".txt", rdfs[i])

        plt.legend(frameon=False)
        plt.xlabel(r"r (nm)")
        plt.ylabel(r"g(r)")
        plt.savefig(f"RDFs_{agg_size}.pdf", bbox_inches="tight")
        plt.show()

    # now save information regarding the Is and Rg
    if compute_i_rg:
        I1 = np.array(I1)
        I2 = np.array(I2)
        I3 = np.array(I3)
        Rg = np.array(Rg)
        Ia = np.mean(I1)
        Ib = np.mean(I2)
        Ic = np.mean(I3)
        Rgm = np.mean(Rg)
        a = np.sqrt((5.0 / (2.0 * total_mass)) * (Ib + Ic - Ia))
        b = np.sqrt((5.0 / (2.0 * total_mass)) * (Ia + Ic - Ib))
        c = np.sqrt((5.0 / (2.0 * total_mass)) * (Ia + Ib - Ic))

        # clear plot and plot principal moment of inertia
        plt.clf()

        plt.plot(I1, label=r"I$_1$")
        plt.axhline(Ia, linestyle="--", color="#1f77b4")
        plt.plot(I2, label=r"I$_2$")
        plt.axhline(Ib, linestyle="--", color="#ff7f0e")
        plt.plot(I3, label=r"I$_3$")
        plt.axhline(Ic, linestyle="--", color="#2ca02c")

        plt.legend(frameon=False)
        plt.xlabel(r"Config")
        plt.ylabel(r"MOI (Da $\AA^2$)")

        plt.savefig(f"Is_{agg_size}.pdf", bbox_inches="tight")
        plt.show()

        # clear plot and plot Rg
        plt.clf()

        plt.plot(Rg)

        plt.xlabel(r"Config")
        plt.ylabel(r"R$_g$ ($\AA$)")
        plt.savefig(f"Rg_{agg_size}.pdf", bbox_inches="tight")
        plt.show()

        # Now print average values and alpha
        with open(f"summary_{agg_size}.dat", "w") as f:
            f.write(
                "I1 = {}±{} Da A^2, I2 = {}±{} Da A^2, I3 = {}±{} Da A^2\n\n".format(
                    Ia, np.std(I1), Ib, np.std(I2), Ic, np.std(I3)
                )
            )
            f.write("a = {} A, b = {} A, c = {} A\n\n".format(a, b, c))

            f.write("alpha = {}\n\n".format((2 * Ia - Ib - Ic) / (Ia + Ib + Ic)))
            f.write("Rg = {}±{} A\n".format(Rgm, np.std(Rg)))

    if compute_pddf:
        mid_bin_pddf = (pddf_edges[:-1] + pddf_edges[1:]) / 2.0

        # pddf /= np.max(pddf)

        np.savetxt(f"{agg_size}_" + "PDDF.txt", pddf)
        np.savetxt(f"{agg_size}_" + "PDDF_bins.txt", mid_bin_pddf)

        plt.plot(mid_bin_pddf / 10.0, pddf, linewidth=3)

        plt.xlabel(r"r (nm)")
        plt.ylabel(r"PDDF")
        plt.savefig(f"PDDF_{agg_size}.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    description = "Given the directory with all the .pdb generated by aggregates.py compute the RDFs for a given size"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("topol", type=str, help=".toml topology files")
    parser.add_argument("pdb_dir", type=str, help="directory containing the .pdb")
    parser.add_argument(
        "agg_size", type=int, help="aggregate size for which the RDFs will be computed"
    )
    parser.add_argument(
        "selections",
        type=str,
        nargs="+",
        help='use quotes to specify the selections for the RDFs (e.g. "name C" "name TC5")',
    )
    parser.add_argument(
        "--name-selections",
        type=str,
        nargs="+",
        help='use quotes to specify the names of selections for the RDFs (e.g. "Phenyl 1" "Phenyl 2")',
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=100,
        help="number of bins used in the histogram (default = 100)",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=10.0,
        help="maximum distance of RDF (defualt = 10.0 A)",
    )
    parser.add_argument(
        "--center",
        type=str,
        default=False,
        help="selection for the center (if not used the center is the centroid)",
    )
    parser.add_argument(
        "--masses",
        type=str,
        default=None,
        help="file containing residue name (molecule name) in first column and a column with the mass for each bead",
    )
    parser.add_argument(
        "--save-centered-aggregate",
        action="store_true",
        default=False,
        help='for each snapshot containing an aggregate of selected size, save the snapshot with the aggregate centered in the "centered" directory',
    )
    parser.add_argument(
        "--save-aggregate-only",
        action="store_true",
        default=False,
        help="for each snapshot, save the configuration of the aggregate only"
    )
    parser.add_argument(
        "--do-not-compute-rdfs",
        action="store_true",
        default=False,
        help="skip RDF computations",
    )
    parser.add_argument(
        "--principal-moments-rg",
        action="store_true",
        default=False,
        help="compute the principal moments of inertia and radius of gyration (requires --masses)",
    )
    parser.add_argument(
        "--compute-pddf",
        action="store_true",
        default=False,
        help="compute the pair distance distribution function (PDDF)",
    )

    args = parser.parse_args()

    if args.principal_moments_rg and not args.masses:
        raise AssertionError("--principal-moments-rg requires passing the masses")
    
    if args.name_selections and (len(args.selections) != len(args.name_selections)):
        raise AssertionError("the number of selections must be equal to the number of names")
    elif not args.name_selections:
        args.name_selections = args.selections

    topol = process_topology(args.topol)

    # write options to file
    with open("summary_compute_rdf_aggregates.txt", "w") as f:
        f.write("Command: " + " ".join(sys.argv) + "\n")

    compute_rdfs(
        topol,
        args.pdb_dir,
        args.agg_size,
        args.selections,
        args.name_selections,
        args.center,
        args.nbins,
        args.rmax,
        parse_masses(args.masses),
        args.do_not_compute_rdfs,
        args.principal_moments_rg,
        args.compute_pddf,
        args.save_centered_aggregate,
        args.save_aggregate_only,
    )
