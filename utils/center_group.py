import argparse
import os
import re
import sys

import h5py
import numpy as np


# based on https://stackoverflow.com/a/6512463/3254658
def parse_bead_list(string):
    m = re.match(r"(\d+)(?:-(\d+))?$", string)
    if not m:
        raise argparse.ArgumentTypeError(
            "'"
            + string
            + "' is not a range of number. Expected forms like '0-5' or '2'."
        )
    start = int(m.group(1), base=10)
    end = int(m.group(2), base=10) or start
    if end < start:
        raise argparse.ArgumentTypeError(
            f"Start value ({start}) should be larger than final value ({end})"
        )
    return list(range(start, end + 1))

def parse_name_list(selection, h5md_file):
    f_in = h5py.File(h5md_file, "r")
    names = f_in["parameters/vmd_structure/name"][:]
    species = f_in["particles/all/species"][:]
    name_to_species = {n: s for s, n in enumerate(names)}

    name_list = np.empty(0, dtype=np.int32)
    for name in selection:
        name_list = np.append(name_list, np.where(species == name_to_species[np.bytes_(name)]))

    f_in.close()
    return name_list


def get_centers(positions, box, nrefs=3):
    centers = np.empty((0, positions.shape[2]))
    # based on the position of the first atom get minimal distances
    for frame in range(positions.shape[0]):
        prevpos = np.copy(positions[frame, :, :])

        # select a few random references to apply minimum image convention
        for _ in range(nrefs):
            refbead = np.random.randint(positions.shape[1])
            mask = np.ones(positions.shape[1], bool)
            mask[refbead] = False

            deltas = prevpos[mask, :] - prevpos[refbead, :]
            subtract = np.where(deltas > 0.5 * np.diag(box[frame]), True, False)
            add = np.where(-deltas > 0.5 * np.diag(box[frame]), True, False)

            newpos = np.where(
                subtract,
                prevpos[mask, :] - np.diag(box[frame]),
                prevpos[mask, :],
            )
            newpos = np.where(add, prevpos[mask, :] + np.diag(box[frame]), newpos[:, :])
            newpos = np.insert(newpos, refbead, prevpos[refbead, :], axis=0)
            prevpos = newpos

        # get the centroid with (hopefully) the minimum image convention
        centers = np.append(centers, [newpos.mean(axis=0)], axis=0)

    return centers


def center_trajectory_mic(h5md_file, bead_list, nrefbeads, out_path, center_last=False):
    f_in = h5py.File(h5md_file, "r")
    f_out = h5py.File(out_path, "w")

    for k, v in f_in.attrs.items():
        f_out.attrs[k] = v

    for k in f_in.keys():
        f_in.copy(k, f_out)

    box_size = f_in["particles/all/box/edges/value"][:]
    beads_pos = f_in["particles/all/position/value"][:][:, bead_list, :]
    centers = get_centers(beads_pos, box_size, nrefbeads)

    box_diag = np.diag(box_size[0])
    for frame in range(1, box_size.shape[0]):
        box_diag = np.vstack((box_diag, np.diag(box_size[frame])))
    translate = (0.5 * box_diag) - centers

    translations = np.repeat(
        translate[:, np.newaxis, :],
        f_in["particles/all/position/value"].shape[1],
        axis=1,
    )

    if center_last:
        tpos = f_in["particles/all/position/value"] + translations[-1]
    else:
        tpos = f_in["particles/all/position/value"] + translations

    f_out["particles/all/position/value"][:] = np.mod(
        tpos, np.repeat(box_diag[:, np.newaxis, :], tpos.shape[1], axis=1)
    )

    f_in.close()
    f_out.close()

def center_of_mass(pos, n, box):
    p_mapped = 2 * np.pi * pos / box
    cos_p_mapped = np.cos(p_mapped)
    sin_p_mapped = np.sin(p_mapped)

    cos_average = np.sum(cos_p_mapped) / n
    sin_average = np.sum(sin_p_mapped) / n

    theta = np.arctan2(-sin_average, -cos_average) + np.pi
    return box * theta / (2 * np.pi)


def get_centers_com(positions, box_size, axis):
    frames = positions.shape[0]
    n = positions.shape[1]
    centers = np.zeros((frames, 3))
    for frame in range(frames):
        centers[frame, axis] = center_of_mass(positions[frame, :, axis], n, box_size[frame, axis])
    return centers


def center_trajectory_com(h5md_file, bead_list, axis, out_path):
    f_in = h5py.File(h5md_file, "r")
    f_out = h5py.File(out_path, "w")

    for k, v in f_in.attrs.items():
        f_out.attrs[k] = v

    for k in f_in.keys():
        f_in.copy(k, f_out)
    f_in.close()

    box_size = f_out["particles/all/box/edges/value"][:]
    beads_pos = f_out["particles/all/position/value"][:][:, bead_list, :]

    box_diag = np.diag(box_size[0])
    for frame in range(1, box_size.shape[0]):
        box_diag = np.vstack((box_diag, np.diag(box_size[frame])))

    centers = get_centers_com(beads_pos, box_diag, axis)

    mask = np.eye(1, 3, k=axis)
    translate = 0.5 * mask * box_diag - centers
    
    translations = np.repeat(
        translate[:, np.newaxis, :],
        f_in["particles/all/position/value"].shape[1],
        axis=1,
    )
    
    tpos = f_out["particles/all/position/value"] + translations

    f_out["particles/all/position/value"][:] = np.mod(
        tpos, np.repeat(box_diag[:, np.newaxis, :], tpos.shape[1], axis=1)
    )
    f_out.close()


if __name__ == "__main__":
    description = (
        "Center geometric center of beads in the box for each frame in a .H5 trajectory"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5md_file", type=str, help="input .H5MD file name")
    parser.add_argument(
        "-b",
        "--beads",
        type=parse_bead_list,
        nargs="+",
        default=None,
        help="bead list to center (e.g.: 1-100 102-150)",
    )
    parser.add_argument(
        "-n",
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Names of bead to center (e.g.: C1 N G2)",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["COM", "MIC"],
        type=str,
        default=None,
        help="Specify the centering method. Available methods:\n"
            "COM: center the box around the center of mass of the given groups along a direction (axis)\n"
            "MIC: center the box around a centroid of the group that assures the minimal image convention",
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=[0, 1, 2],
        default=2,
        required='COM' in sys.argv,
        help="Direction along which to calculate the center of mass."
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        dest="out_path",
        metavar="file name",
        help="output hymd HDF5 file name",
    )
    parser.add_argument(
        "-f",
        action="store_true",
        default=False,
        dest="force",
        help="overwrite existing output file",
    )
    parser.add_argument(
        "--reference-beads",
        type=int,
        default=3,
        dest="nrefs",
        help="number of reference beads considered when applying the minimum image convention (default=3)",
    )
    parser.add_argument(
        "--center-wrt-last-frame",
        action="store_true",
        default=False,
        dest="center_last",
        help="instead of centering the groups for every frame, translate all frames with respect to the center of the last frame",
    )
    args = parser.parse_args()

    if args.beads is None and args.names is None:
        error_str = "Either the 'beads' or 'names' variable must be provided."
        raise ValueError(error_str)

    if args.out_path is None:
        args.out_path = os.path.join(
            os.path.abspath(os.path.dirname(args.h5md_file)),
            os.path.splitext(os.path.split(args.h5md_file)[-1])[0]
            + "_new"
            + os.path.splitext(os.path.split(args.h5md_file)[-1])[1],
        )
    if os.path.exists(args.out_path) and not args.force:
        error_str = (
            f"The specified output file {args.out_path} already exists. "
            f'use overwrite=True ("-f" flag) to overwrite.'
        )
        raise FileExistsError(error_str)

    if args.beads is not None:
        bead_list = []
        for interval in args.beads:
            bead_list += interval

        bead_list = np.array(sorted(bead_list)) - 1

    if args.names is not None:
       name_list = parse_name_list(args.names, args.h5md_file)

    if args.beads is not None and args.names is not None:
        atom_list = np.intersect1d(bead_list, name_list)
    elif args.beads is not None:
        atom_list = bead_list
    else:
        atom_list = name_list

    if args.method == "COM":
        center_trajectory_com(
            args.h5md_file, 
            atom_list, 
            args.axis, 
            args.out_path,
        )

    if args.method == "MIC":
        center_trajectory_mic(
            args.h5md_file,
            atom_list,
            nrefbeads=args.nrefs,
            out_path=args.out_path,
            center_last=args.center_last,
        )

