import os
import argparse
import h5py
import numpy as np
import re


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


def get_centers(positions, box, nrefs=3):
    centers = np.empty((0, positions.shape[2]))
    # based on the position of the first atom get minimal distances
    for frame in range(positions.shape[0]):
        prevpos = np.copy(positions[frame,:,:])
        
        # select a few random references to apply minimal image convention
        for i in range(nrefs):
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
            newpos = np.where(
                add, prevpos[mask, :] + np.diag(box[frame]), newpos[:, :]
            )
            newpos = np.insert(newpos, refbead, prevpos[refbead, :], axis=0)
            prevpos = newpos

        # get the centroid with (hopefully) the minimal image convention
        centers = np.append(centers, [newpos.mean(axis=0)], axis=0)

    return centers


def center_trajectory(h5md_file, bead_list, overwrite=False, out_path=None):
    if out_path is None:
        out_path = os.path.join(
            os.path.abspath(os.path.dirname(h5md_file)),
            os.path.splitext(os.path.split(h5md_file)[-1])[0]
            + "_new"
            + os.path.splitext(os.path.split(h5md_file)[-1])[1],
        )
    if os.path.exists(out_path) and not overwrite:
        error_str = (
            f"The specified output file {out_path} already exists. "
            f'use overwrite=True ("-f" flag) to overwrite.'
        )
        raise FileExistsError(error_str)

    f_in = h5py.File(h5md_file, "r")
    f_out = h5py.File(out_path, "w")

    for k, v in f_in.attrs.items():
        f_out.attrs[k] = v

    for k in f_in.keys():
        f_in.copy(k, f_out)

    box_size = f_in["particles/all/box/edges/value"][:]

    beads_pos = f_in["particles/all/position/value"][:][:, bead_list, :]
    centers = get_centers(beads_pos, box_size)

    box_diag = np.diag(box_size[0])
    for frame in range(1, box_size.shape[0]):
        box_diag = np.vstack((box_diag, np.diag(box_size[frame])))
    translate = (0.5 * box_diag) - centers

    translations = np.repeat(
        translate[:, np.newaxis, :],
        f_in["particles/all/position/value"].shape[1],
        axis=1,
    )

    tpos = f_in["particles/all/position/value"] + translations
    f_out["particles/all/position/value"][:] = np.mod(
        tpos, np.repeat(box_diag[:, np.newaxis, :], tpos.shape[1], axis=1)
    )

    f_in.close()
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
        required=True,
        help="bead list to center (e.g.: 1-100 102-150)",
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
    args = parser.parse_args()

    bead_list = []
    for interval in args.beads:
        bead_list += interval

    bead_list = np.array(sorted(bead_list)) - 1

    center_trajectory(
        args.h5md_file, bead_list, overwrite=args.force, out_path=args.out_path
    )
