import MDAnalysis as mda
from MDAnalysis.lib import distances, mdamath, transformations
from scipy.spatial.distance import squareform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
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


def parse_parametersfile(fname):
    params = []
    with open(fname, "r") as f:
        for line in f:
            if "PARAMETERS" in line:
                right_line = line.split("=")[1].strip()
                params.append(float(right_line.split(",")[0])) # get only the first parameter even for SAXS
    return params


def compute_rdfs(
    topol,
    pdbdir,
    agg_size,
    selections,
    name_selections,
    center,
    nbins,
    nbinsl,
    rmax,
    lmax,
    natoms_normalization,
    masses,
    skip_rdfs,
    compute_i_rg,
    compute_pddf,
    compute_cdfs,
    save_centered,
    save_agg_only,
    names_prefix,
    start_colorbar,
    width_colorbar,
    fig_size=(10, 8),
    fig_size_cdf=(12,12),
):
    # initialize
    rdfs = {}
    cdfs = {}
    for i in range(len(selections)):
        rdfs[i] = np.zeros(nbins)
        cdfs[i] = np.zeros((nbinsl, nbins))
    if compute_i_rg:
        I1 = []
        I2 = []
        I3 = []
        Rg = []
    if compute_pddf:
        pddf = np.zeros(nbins)
        
        # read the form factors / scattering lengths
        form_factors = parse_parametersfile(compute_pddf)

    if not names_prefix:
        names_prefix = ""

    nsnaps = 0
    for snapshot in tqdm(glob(os.path.join(pdbdir, "*.pdb"))):
        u = mda.Universe(snapshot)

        if not natoms_normalization:
            natoms_normalization = len(u.atoms)


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
            if save_agg_only or save_centered or compute_i_rg or compute_cdfs:
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

            # set the masses
            if masses:
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

            # compute the principal moment of inertia and Rg
            if compute_i_rg:
                agg_sel = u.select_atoms(f"resid {resid}")

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
                # sqdistmat = squareform(cond_distmat)

                # Create a 2D array of form factors for all pairs of atoms
                form_factors_array = np.array([form_factors[atom.index] for atom in agg_sel])

                # Compute the weights for all pairs of atoms
                weights_matrix = np.outer(form_factors_array, form_factors_array)

                # Get the upper triangular part of the weights matrix
                weights = weights_matrix[np.triu_indices(n, k=1)]

                # Compute the histogram
                pddf_i, pddf_edges = np.histogram(cond_distmat, bins=nbins, range=(0, rmax), weights=weights, density=False)

                pddf += pddf_i

            # compute the cylindrical distribution function
            if compute_cdfs:
                # Based on https://github.com/RichardMandle/cylindr
                # Cite: 10.1371/journal.pone.0279679

                # first align the principal axis of the aggregate with the z-axis
                agg_sel = u.select_atoms(f"resid {resid}")

                # get principal axis
                pa = agg_sel.principal_axes()[2]

                # get the angle between the principal axis and the z-axis and rotate universe
                angle = np.degrees(mdamath.angle(pa, [0,0,1]))
                ax = transformations.rotaxis(pa, [0,0,1])
                u.atoms.rotateby(angle, ax, point=cog)
                # dirname = f"./rotated_cdf_{agg_size}"

                # if not os.path.exists(dirname):
                #     os.mkdir(dirname)

                # u.atoms.write(
                #     os.path.join(dirname, f"rotated_{os.path.basename(snapshot)}")
                # )

                # create selections for which CDFs will be computed
                for i, sel_string in enumerate(selections):
                    if sel_string.strip().lower() in ["name w", "type w", "name na", "type na", "name cl", "type cl"]:
                        at_sel = u.select_atoms(sel_string)
                    else:
                        at_sel = u.select_atoms(f"resid {resid} and " + sel_string)


                    # Compute the distances between the target atoms and the center
                    distances_matrix = at_sel.positions - cog

                    # create array with z and r distances
                    zdist = distances_matrix[:, 2]
                    rdist = np.sqrt(distances_matrix[:, 0] ** 2 + distances_matrix[:, 1] ** 2)

                    # Exclude positions where z is greater than lmax or smaller than -lmax
                    valid_indices = np.where((zdist >= -lmax) & (zdist <= lmax))
                    zdist = zdist[valid_indices]
                    rdist = rdist[valid_indices]

                    # Compute the CDF
                    cdf, yedges, xedges = np.histogram2d(
                        zdist, rdist, bins=[nbinsl, nbins], range=[[-lmax, lmax], [0, rmax]], density=False
                    )
                    cdfs[i] += cdf * np.prod(box_vectors[:3]) / natoms_normalization

                    # Compute the RDF using numpy histogram
                    rdf, bin_edges = np.histogram(
                        rdist, bins=nbins, range=(0, rmax), density=False
                    )
                    rdfs[i] += rdf * np.prod(box_vectors[:3]) / natoms_normalization


            elif not skip_rdfs:
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

                    rdfs[i] += rdf * np.prod(box_vectors[:3]) / natoms_normalization


    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 22

    if compute_cdfs:
        # compute the volume of each shell
        # shell_volumes = np.zeros((np.size(yedges) - 1, np.size(xedges) - 1))
        # for n in range(0, np.size(yedges) -1):
        #     for m in range(0, np.size(xedges) - 1):
        #         shell_volumes[n, m] = ((2 * np.pi * (xedges[m + 1]) ** 2) - (2 * np.pi * (xedges[m - 1]) ** 2)) * np.abs(yedges[2] - yedges[1])
        shell_volumes = np.pi * (xedges[1:] ** 2 - xedges[:-1] ** 2) * np.abs(yedges[1] - yedges[0])
        xedges /= 10.0
        yedges /= 10.0
        # prepare to plot
        xcenters = (xedges[:-1] + xedges[1:]) / 2.0
        ycenters = (yedges[:-1] + yedges[1:]) / 2.0
        np.savetxt(names_prefix + f"{agg_size}_binsR.txt", xcenters)
        np.savetxt(names_prefix + f"{agg_size}_binsL.txt", ycenters)

        # normalize CDFs and find the lower and largest values
        minval = np.inf
        maxval = -np.inf
        for i in range(len(selections)):
            cdfs[i] = cdfs[i] / (shell_volumes * nsnaps)
            if minval > np.min(cdfs[i]):
                minval = np.min(cdfs[i])
            if maxval < np.max(cdfs[i]):
                maxval = np.max(cdfs[i])

        # normalize the CDF by dividing by the shell volume and number of snapshots and plot
        fig, ax = plt.subplots(nrows=2, ncols=int(np.ceil(len(selections)/2)), figsize=fig_size_cdf)
        fig.tight_layout(w_pad=0.4, h_pad=1.8)
        for a in ax.flatten():
            a.set_aspect(5/10)
        if int(np.ceil(len(selections)/2)) == 1:
            for i, sel in enumerate(selections):
                ax[i].set_title(name_selections[i])
                ax[i].set_xlim(xedges[[0, -1]])
                ax[i].set_ylim(yedges[[0, -1]])
                ax[i].set_xticks(np.linspace(xedges[0], xedges[-1], 3))
                ax[i].set_yticks(np.linspace(yedges[0], yedges[-1], 7))
                np.savetxt(names_prefix + f"CDFs_{agg_size}_" + sel.replace(" ", "_") + ".txt", cdfs[i])
                im = ax[i].imshow(cdfs[i], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', interpolation='bilinear', vmin=minval, vmax=maxval, cmap=plt.cm.RdBu)
                ax[i].grid()
                ax[i].set_xlabel(r"r (nm)")
                ax[i].set_ylabel(r"h (nm)")
        else:
            # make the last plot blank if the number of selections is odd
            if len(selections) % 2 != 0:
                ax[-1][-1].axis('off')
            for i, sel in enumerate(selections):
                idx1, idx2 = i % 2, int(i / 2)
                ax[idx1][idx2].set_title(name_selections[i])
                ax[idx1][idx2].set_xlim(xedges[[0, -1]])
                ax[idx1][idx2].set_ylim(yedges[[0, -1]])
                ax[idx1][idx2].set_xticks(np.linspace(xedges[0], xedges[-1], 3))
                ax[idx1][idx2].set_yticks(np.linspace(yedges[0], yedges[-1], 7))
                np.savetxt(names_prefix + f"CDFs_{agg_size}_" + sel.replace(" ", "_") + ".txt", cdfs[i])
                im = ax[idx1][idx2].imshow(cdfs[i], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', interpolation='bilinear', vmin=minval, vmax=maxval, cmap=plt.cm.RdBu)
                ax[idx1][idx2].grid()
                ax[idx1][idx2].set_xlabel(r"r (nm)")
                ax[idx1][idx2].set_ylabel(r"h (nm)")
            # Add color bar
        cax = fig.add_axes([start_colorbar, 0.15, width_colorbar, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('CDF')
        fig.subplots_adjust(hspace=0.35, right=start_colorbar)
        plt.savefig(names_prefix + f"CDFs_{agg_size}.pdf")
        plt.show()
        plt.close('all')

        # radial
        # shell_volumes = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2) * box_vectors[2]
        shell_volumes = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2) * 2.0 * lmax
        # Prepare to plot
        mid_bin = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        np.savetxt(names_prefix + f"gperp_{agg_size}_bins.txt", mid_bin)

        # Normalize the RDF by dividing by the shell volume and number of snapshots and plot
        for i, sel in enumerate(selections):
            rdfs[i] = rdfs[i] / (shell_volumes * nsnaps)

            plt.plot(mid_bin / 10.0, rdfs[i], linewidth=3, label=name_selections[i])

            np.savetxt(names_prefix + f"gperp_{agg_size}_" + sel.replace(" ", "_") + ".txt", rdfs[i])

        plt.legend(frameon=False)
        plt.xlabel(r"r (nm)")
        plt.ylabel(r"g$_\perp$(r)")
        plt.savefig(names_prefix + f"radial_RDFs_{agg_size}.pdf", bbox_inches="tight")
        plt.show()

    # check if rdfs are skipped
    elif not skip_rdfs:
        # Compute the average volume of each shell
        shell_volumes = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

        # Prepare to plot
        mid_bin = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        np.savetxt(names_prefix + f"{agg_size}_bins.txt", mid_bin)

        # Normalize the RDF by dividing by the shell volume and number of snapshots and plot
        for i, sel in enumerate(selections):
            rdfs[i] = rdfs[i] / (shell_volumes * nsnaps)

            plt.plot(mid_bin / 10.0, rdfs[i], linewidth=3, label=name_selections[i])

            np.savetxt(names_prefix + f"{agg_size}_" + sel.replace(" ", "_") + ".txt", rdfs[i])

        plt.legend(frameon=False)
        plt.xlabel(r"r (nm)")
        plt.ylabel(r"g(r)")
        plt.savefig(names_prefix + f"RDFs_{agg_size}.pdf", bbox_inches="tight")
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

        plt.savefig(names_prefix + f"Is_{agg_size}.pdf", bbox_inches="tight")
        plt.show()

        # clear plot and plot Rg
        plt.clf()

        plt.plot(Rg)

        plt.xlabel(r"Config")
        plt.ylabel(r"R$_g$ ($\AA$)")
        plt.savefig(names_prefix + f"Rg_{agg_size}.pdf", bbox_inches="tight")
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

        np.savetxt(names_prefix + f"{agg_size}_" + "PDDF.txt", pddf)
        np.savetxt(names_prefix + f"{agg_size}_" + "PDDF_bins.txt", mid_bin_pddf)

        plt.plot(mid_bin_pddf / 10.0, pddf, linewidth=3)

        plt.xlabel(r"r (nm)")
        plt.ylabel(r"PDDF")
        plt.savefig(names_prefix + f"PDDF_{agg_size}.pdf", bbox_inches="tight")
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
        default=50,
        help="number of bins used in the histogram (default = 50)",
    )
    parser.add_argument(
        "--nbinsl",
        type=int,
        default=300,
        help="number of bins used in the histogram for CDF in z direction (default = 300)",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=10.0,
        help="maximum distance of RDF (default = 10.0 A)",
    )
    parser.add_argument(
        "--lmax",
        type=float,
        default=30.0,
        help="maximum distance for CDF z direction (default = 30.0 A)",
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
        "--compute-cdfs",
        action="store_true",
        default=False,
        help="instead of computing RDFs, compute CDFs (requires --masses)",
    )
    parser.add_argument(
        "--principal-moments-rg",
        action="store_true",
        default=False,
        help="compute the principal moments of inertia and radius of gyration (requires --masses)",
    )
    parser.add_argument(
        "--compute-pddf",
        type=str,
        default=None,
        help="compute the pair distance distribution function (PDDF) - needs parameters file just like for PLUMED",
    )
    parser.add_argument(
        "--natoms-normalization",
        type=int,
        default=None,
        help="number of atoms to normalize the RDFs/CDFs",
    )
    parser.add_argument(
        "--names-prefix",
        type=str,
        default=None,
        help="prefix for the names of the saved files",
    )
    parser.add_argument(
        "--fig-size",
        type=int,
        nargs=2,
        default=(10, 8),
        help="two integers to define the size of the RDF figure (default = 10 8)",
    )
    parser.add_argument(
        "--fig-size-cdf",
        type=int,
        nargs=2,
        default=(12, 12),
        help="two integers to define the size of the CDF figure (default = 12 12)",
    )
    parser.add_argument(
        "--start-colorbar",
        type=float,
        default=0.85,
        help="value for which after that the colorbar will start (default = 0.85) in percentage of width",
    )
    parser.add_argument(
        "--width-colorbar",
        type=float,
        default=0.02,
        help="width of the colorbar (default = 0.02)",
    )

    args = parser.parse_args()

    if args.principal_moments_rg and not args.masses:
        raise AssertionError("--principal-moments-rg requires passing the masses")
    if args.compute_cdfs and not args.masses:
        raise AssertionError("--compute-cdfs requires passing the masses")
    
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
        args.nbinsl,
        args.rmax,
        args.lmax,
        args.natoms_normalization,
        parse_masses(args.masses),
        args.do_not_compute_rdfs,
        args.principal_moments_rg,
        args.compute_pddf,
        args.compute_cdfs,
        args.save_centered_aggregate,
        args.save_aggregate_only,
        args.names_prefix,
        args.start_colorbar,
        args.width_colorbar,
        tuple(args.fig_size),
        tuple(args.fig_size_cdf),
    )
