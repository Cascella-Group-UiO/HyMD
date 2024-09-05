import MDAnalysis as mda
from MDAnalysis.lib import distances, mdamath, transformations
from glob import glob
import argparse
from tqdm import tqdm
import os
import numpy as np
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



def compute_order_parameter(topol, pdbdir, agg_size, core_selection, masses, center, eccentricity_threshold=0.14):
    type_map = {"sphere": 0, "oblate": 1, "prolate": 2}
    agg_types = []
    I1 = {0: [], 1: [], 2: []}
    I2 = {0: [], 1: [], 2: []}
    I3 = {0: [], 1: [], 2: []}

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

        # for each aggregate of given size, get the order parameter
        for ididx, resid in enumerate(agg_resids):
            if ididx > 0:
                u = mda.Universe(snapshot)
            nsnaps += 1
            box_vectors = u.dimensions
            agg_sel = u.select_atoms(f"resid {resid}")

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

            box_center = box_vectors[:3] / 2.0
            u.atoms.translate(box_center - cog)
            u.atoms.wrap(compound="atoms")
            cog = box_center

            # set the masses
            i = 0
            mass_array = []
            while i < len(agg_sel.resnames):
                name = agg_sel.resnames[i]
                for bmass in masses[name]:
                    mass_array.append(bmass)
                i += len(masses[name])

            mass_array = np.array(mass_array)
            agg_sel.masses = mass_array

            # get moment of inertia and principal axis
            I = agg_sel.moment_of_inertia()
            UT = agg_sel.principal_axes()  # this is already transposed

            # diagonalizes I
            Lambda = UT.dot(I.dot(UT.T))

            Ia = Lambda[0][0]
            Ib = Lambda[1][1]
            Ic = Lambda[2][2]

            # detect the aggregate type
            type_agg = None
            # compute eccentricity
            e = 1.0 - np.min([Ia, Ib, Ic]) / np.mean([Ia, Ib, Ic])
            if e < eccentricity_threshold:
                type_agg = type_map["sphere"]
            else:
                # Ic is the lowest value (biggest axis)
                if np.abs(Ia - Ib) > np.abs(Ib - Ic): 
                    continue # if oblate, skip
                    # type_agg = type_map["oblate"]
                else:
                    type_agg = type_map["prolate"]
            agg_types.append(type_agg)

            # gets symmetry axis
            if type_agg == type_map["oblate"]:
                pa = UT[0]
            else:
                pa = UT[2]

            # get the angle between the principal axis and the z-axis and rotate universe
            # this does not take into consideration prolate or oblate micelles
            angle = np.degrees(mdamath.angle(pa, [0,0,1]))
            ax = transformations.rotaxis(pa, [0,0,1])
            u.atoms.rotateby(angle, ax, point=cog)

            # select the core and get dimensions
            core = u.select_atoms(f"resid {resid} and {core_selection}")
            total_mass = np.sum(core.masses)

            # get moment of inertia and principal axis
            I = core.moment_of_inertia()
            UT = core.principal_axes()  # this is already transposed

            # diagonalizes I
            Lambda = UT.dot(I.dot(UT.T))

            I1[type_agg].append(Lambda[0][0])
            I2[type_agg].append(Lambda[1][1])
            I3[type_agg].append(Lambda[2][2])

    # for each aggregate type, compute the dimensions
    for k, v in type_map.items():
        if len(I1[v]) == 0:
            continue
            
        Ia = np.mean(I1[v])
        Ib = np.mean(I2[v])
        Ic = np.mean(I3[v])

        a = np.sqrt((5.0 / (2.0 * total_mass)) * (Ib + Ic - Ia))
        b = np.sqrt((5.0 / (2.0 * total_mass)) * (Ia + Ic - Ib))
        c = np.sqrt((5.0 / (2.0 * total_mass)) * (Ia + Ib - Ic))

        print(f"Aggregate type: {k}")
        print(f"Dimensions: a = {a:.4f} A, b = {b:.4f} A, c = {c:.4f} A")


if __name__ == "__main__":
    description = "Given the directory with all the .pdb generated by aggregates.py compute the dimensions for the core of the aggregates."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("topol", type=str, help=".toml topology files")
    parser.add_argument("pdb_dir", type=str, help="directory containing the .pdb")
    parser.add_argument(
        "agg_size", type=int, help="aggregate size for which the RDFs will be computed"
    )
    parser.add_argument(
        "core_selection",
        type=str,
        help='Selection for the core beads. Example: "name C2 TC5"',
    )
    parser.add_argument(
        "masses",
        type=str,
        default=None,
        help="file containing residue name (molecule name) in first column and a column with the mass for each bead",
    )
    parser.add_argument(
        "--eccentricity-threshold",
        type=float,
        default=0.14,
        help="eccentricity threshold to determine the type of aggregate",
    )
    parser.add_argument(
        "--center",
        type=str,
        default=False,
        help="selection for the center (if not used the center is the centroid)",
    )
    args = parser.parse_args()
    topol = process_topology(args.topol)
    compute_order_parameter(topol, args.pdb_dir, args.agg_size, args.core_selection, parse_masses(args.masses), args.center, args.eccentricity_threshold)