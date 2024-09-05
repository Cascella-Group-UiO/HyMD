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


def get_index_beads(topol, bead1, bead2):
    for i, atom in enumerate(topol["atoms"]):
        if atom[1] == bead1:
            bead1_index = i
        elif atom[1] == bead2:
            bead2_index = i
    return bead1_index, bead2_index


def compute_order_parameter(topol, pdbdir, agg_size, bead1, bead2, masses, cylinder_length, center, eccentricity_threshold=0.14):
    type_map = {"sphere": 0, "oblate": 1, "prolate": 2}
    agg_types = []
    all_order_parameters = []
    all_cosines = []
    all_order_parameters_cylinder = []
    all_cosines_cylinder = []
    all_order_parameters_cap = []
    all_cosines_cap = []
    all_distances = []
    distances_sphere = []
    distances_prolate = []
    all_angles = []
    angles_sphere = []
    angles_prolate = []
    angles_cap = []

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
            total_mass = np.sum(mass_array)

            # get moment of inertia and principal axis
            I = agg_sel.moment_of_inertia()
            UT = agg_sel.principal_axes()  # this is already transposed

            # diagonalizes I
            Lambda = UT.dot(I.dot(UT.T))

            I1 = Lambda[0][0]
            I2 = Lambda[1][1]
            I3 = Lambda[2][2]

            # detect the aggregate type
            type_agg = None
            # compute eccentricity
            e = 1.0 - np.min([I1, I2, I3]) / np.mean([I1, I2, I3])
            if e < eccentricity_threshold:
                type_agg = type_map["sphere"]
            else:
                # I3 is the lowest value (biggest axis)
                if np.abs(I1 - I2) > np.abs(I2 - I3): 
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

            # iterate over all molecules computing the order parameter
            molecule_order_parameters = []
            simple_cosine = []
            cylinder_order_parameters = []
            cylinder_cosine = []
            cap_order_parameters = []
            cap_cosine = []
            for mol in agg_sel.residues:
                nmols_residue = int(mol.atoms.positions.shape[0] / topol[mol.resname]["atomnum"])
                for i in range(nmols_residue):
                    mol_at_positions = mol.atoms.positions[i * topol[mol.resname]["atomnum"] : (i + 1) * topol[mol.resname]["atomnum"]] - cog
                    # get positions of bead only
                    bead1_index, bead2_index = get_index_beads(topol[mol.resname], bead1, bead2)                  
                    orient_vec = mol_at_positions[bead2_index] - mol_at_positions[bead1_index]
                    all_distances.append(np.linalg.norm(orient_vec))

                    if type_agg == type_map["sphere"]:
                        distances_sphere.append(np.linalg.norm(orient_vec))
                        cm_vec = mol_at_positions[bead1_index]
                        
                        # compute order parameter
                        cosine = np.dot(cm_vec, orient_vec) / (np.linalg.norm(cm_vec) * np.linalg.norm(orient_vec))
                        molecule_order_parameters.append(0.5 * (3.0 * cosine * cosine - 1.0))
                        simple_cosine.append(cosine)
                        angles_sphere.append(np.arccos(cosine) * 180.0 / np.pi)

                    elif type_agg == type_map["prolate"]:
                        distances_prolate.append(np.linalg.norm(orient_vec))
                        # find if the molecule is in the cylindrical or cap regions
                        if mol_at_positions[bead1_index][2] > cylinder_length:  # top cap
                            cm_vec = mol_at_positions[bead1_index]
                            cm_vec[2] -= cylinder_length

                            cosine = np.dot(cm_vec, orient_vec) / (np.linalg.norm(cm_vec) * np.linalg.norm(orient_vec))
                            cap_order_parameters.append(0.5 * (3.0 * cosine * cosine - 1.0))
                            cap_cosine.append(cosine)
                            angles_cap.append(np.arccos(cosine) * 180.0 / np.pi)
                        elif mol_at_positions[bead1_index][2] < -cylinder_length:  # bottom cap
                            cm_vec = mol_at_positions[bead1_index]
                            cm_vec[2] += cylinder_length

                            cosine = np.dot(cm_vec, orient_vec) / (np.linalg.norm(cm_vec) * np.linalg.norm(orient_vec))
                            cap_order_parameters.append(0.5 * (3.0 * cosine * cosine - 1.0))
                            cap_cosine.append(cosine)
                            angles_cap.append(np.arccos(cosine) * 180.0 / np.pi)
                        else:  # cylinder part
                            cm_vec = mol_at_positions[bead1_index]
                            cm_vec[2] = 0.0

                            cosine = np.dot(cm_vec, orient_vec) / (np.linalg.norm(cm_vec) * np.linalg.norm(orient_vec))
                            cylinder_order_parameters.append(0.5 * (3.0 * cosine * cosine - 1.0))
                            cylinder_cosine.append(cosine)
                            angles_prolate.append(np.arccos(cosine) * 180.0 / np.pi)


                    all_angles.append(np.arccos(cosine) * 180.0 / np.pi)

            if type_agg == type_map["sphere"]:
                all_order_parameters.append(np.mean(molecule_order_parameters))
                all_cosines.append(np.mean(simple_cosine))
            elif type_agg == type_map["prolate"]:
                all_order_parameters_cylinder.append(np.mean(cylinder_order_parameters))
                all_cosines_cylinder.append(np.mean(cylinder_cosine))
                all_order_parameters_cap.append(np.mean(cap_order_parameters))
                all_cosines_cap.append(np.mean(cap_cosine))

    print(f"Aggregate size: {agg_size}")
    print(f"Eccentricity threshold: {eccentricity_threshold}")
    print(f"Prolate cylinder half length: {cylinder_length}\n")

    print(f"Percentage of spheres: {agg_types.count(type_map['sphere']) / nsnaps}")
    print(f"Percentage of oblates: {agg_types.count(type_map['oblate']) / nsnaps}")
    print(f"Percentage of prolates: {agg_types.count(type_map['prolate']) / nsnaps}\n")

    print(f"Mean distance between beads: {np.mean(all_distances)} ± {np.std(all_distances)}")
    print(f"Mean distance between beads (sphere): {np.mean(distances_sphere)} ± {np.std(distances_sphere)}")
    print(f"Mean distance between beads (prolate): {np.mean(distances_prolate)} ± {np.std(distances_prolate)}\n")

    print(f"Mean order parameter (sphere): {np.mean(all_order_parameters)} ± {np.std(all_order_parameters)}")
    print(f"Mean cosine (sphere): {np.mean(all_cosines)} ± {np.std(all_cosines)}")
    print(f"Mean order parameter (cylinder): {np.mean(all_order_parameters_cylinder)} ± {np.std(all_order_parameters_cylinder)}")
    print(f"Mean cosine (cylinder): {np.mean(all_cosines_cylinder)} ± {np.std(all_cosines_cylinder)}")
    print(f"Mean order parameter (cap): {np.mean(all_order_parameters_cap)} ± {np.std(all_order_parameters_cap)}")
    print(f"Mean cosine (cap): {np.mean(all_cosines_cap)} ± {np.std(all_cosines_cap)}\n")

    print(f"Mean angle (all): {np.mean(all_angles)} ± {np.std(all_angles)}")

    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["font.size"] = 22

    # Plotting distributions
    angles = [all_angles, angles_cap, angles_prolate, angles_sphere]
    labels = ['All Angles', 'Cap Angles', 'Prolate Angles', 'Sphere Angles']
    colors = ['C0', 'C1', 'C2', 'C3']
    counts = [len(angle_list) for angle_list in angles]
    sorted_angles = [angle_list for _, angle_list in sorted(zip(counts, angles), reverse=True)]
    sorted_labels = [label for _, label in sorted(zip(counts, labels), reverse=True)]
    sorted_colors = [color for _, color in sorted(zip(counts, colors), reverse=True)]
    
    for angle_list, label, color in zip(sorted_angles, sorted_labels, sorted_colors):
        plt.hist(angle_list, bins=50, alpha=0.5, label=label, color=color)

    # Adding labels and legend
    plt.xlabel(r'Angle ($\degree$)')
    plt.ylabel('Frequency')
    plt.legend()

    # Displaying the plot
    plt.savefig(f"distr_angles_surface_{agg_size}.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    description = "Given the directory with all the .pdb generated by aggregates.py compute the order parameters for each aggregate type"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("topol", type=str, help=".toml topology files")
    parser.add_argument("pdb_dir", type=str, help="directory containing the .pdb")
    parser.add_argument(
        "agg_size", type=int, help="aggregate size for which the RDFs will be computed"
    )
    parser.add_argument(
        "bead1",
        type=str,
        help='name of first bead in the order parameter calculation. Example: "TN2a"',
    )
    parser.add_argument(
        "bead2",
        type=str,
        help='name of second bead in the order parameter calculation. Example: "TP1"',
    )
    parser.add_argument(
        "masses",
        type=str,
        default=None,
        help="file containing residue name (molecule name) in first column and a column with the mass for each bead",
    )
    parser.add_argument(
        "--cylinder-length",
        type=float,
        default=0.0,
        help="HALF length of the cylinder (not considering the cap) to compute the order parameter",
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

    compute_order_parameter(topol, args.pdb_dir, args.agg_size, args.bead1, args.bead2, parse_masses(args.masses), args.cylinder_length, args.center, args.eccentricity_threshold)

