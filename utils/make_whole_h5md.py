import h5py
import tqdm
import argparse
import numpy as np


def make_whole(
    file_path, hymd_inp, out_path=None, frames=None, scale=None, shift=None,
):
    if out_path is None:
        out_path = (
            "".join(file_path.split(".")[:-1])
            + "_whole."
            + file_path.split(".")[-1]
        )

    with h5py.File(out_path, "w") as out_file:
        with h5py.File(hymd_inp, "r") as inp_file:
            with h5py.File(file_path, "r") as in_file:

                class Copy:
                    def __init__(self, source, dest):
                        self.source = source
                        self.dest = dest

                    def __call__(self, name):
                        if isinstance(self.source[name], h5py.Group):
                            # print(f"Copying Group   -> {name}")
                            self.dest["/"].create_group(name)

                        elif isinstance(self.source[name], h5py.Dataset):
                            # print(f"Copying Dataset -> {name}")
                            self.dest.create_dataset(
                                name, shape=self.source[name].shape,
                                dtype=self.source[name].dtype,
                                data=self.source[name][...],
                            )

                visit_func = Copy(in_file, out_file)
                in_file["/"].visit(visit_func)

                molecules = inp_file["/molecules"][...]

        n_frames = out_file["/particles/all/position/value"].shape[0]
        # n_particles = out_file["/particles/all/position/value"].shape[1]
        # n_dimensions = out_file["/particles/all/position/value"].shape[2]
        box_size = out_file["/particles/all/box/edges"][...]
        bond_from = out_file["/parameters/vmd_structure/bond_from"][...]
        bond_to = out_file["/parameters/vmd_structure/bond_to"][...]

        bond_sort_ind = np.argsort(bond_from)

        out_file["/parameters/vmd_structure/bond_from"][...] = bond_from[bond_sort_ind]  # noqa: E501
        out_file["/parameters/vmd_structure/bond_to"][...] = bond_to[bond_sort_ind]  # noqa: E501

        if ":" in frames[0]:
            from_ = frames[0].split(":")[0]
            to_ = frames[0].split(":")[1]
            frames = list(np.arange(int(from_), int(to_)))
            print(frames)
        try:
            frames_ = [int(f) for f in frames]
            frames = frames_

        except (TypeError, ValueError,):
            if frames[0].lower() == "all":
                frames = [i for i in range(n_frames)]

        print(f"Making frames whole whole: {frames}")

        if shift is not None:
            out_file["/particles/all/position/value"][..., 0] = (
                out_file["/particles/all/position/value"][..., 0] - shift[0]
            )
            out_file["/particles/all/position/value"][..., 1] = (
                out_file["/particles/all/position/value"][..., 1] - shift[1]
            )
            out_file["/particles/all/position/value"][..., 2] = (
                out_file["/particles/all/position/value"][..., 2] - shift[2]
            )
            out_file["/particles/all/position/value"][..., :] = (
                np.mod(out_file["/particles/all/position/value"][..., :], box_size[None, :])
            )

        bond_from = out_file["/parameters/vmd_structure/bond_from"][...] - 1
        bond_to = out_file["/parameters/vmd_structure/bond_to"][...] - 1

        for frame in tqdm.tqdm(frames):
            positions = out_file["/particles/all/position/value"]
            for molecule_index in np.unique(molecules):
                particle_indices = np.where(molecules == molecule_index)
                for particle in particle_indices[0]:
                    r = positions[frame, particle, :]
                    for dim in range(3):
                        if r[dim] > box_size[dim]:
                            positions[frame, particle, dim] -= box_size[dim]
                        if r[dim] < 0.0:
                            positions[frame, particle, dim] += box_size[dim]

                for particle in particle_indices[0]:
                    bonds = bond_to[np.where(bond_from == particle)[0]]
                    for b in bonds:
                        for dim in range(3):
                            ri = positions[frame, particle, dim]
                            rj = positions[frame, b, dim]
                            dr = rj - ri
                            if dr > 0.5 * box_size[dim]:
                                positions[frame, b, dim] -= box_size[dim]
                            if dr <= -0.5 * box_size[dim]:
                                positions[frame, b, dim] += box_size[dim]

        if scale is not None:
            out_file["/particles/all/position/value"][...] = (
                scale * out_file["/particles/all/position/value"][...]
            )
            out_file["/particles/all/box/edges"][...] = (
                scale * out_file["/particles/all/box/edges"][...]
            )
            out_file["/particles/all/box/edges"][...] = (
                scale * out_file["/particles/all/box/edges"][...]
            )


if __name__ == '__main__':
    description = "Make molecules whole in .H5MD files"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "H5MD_file", type=str, help=".H5MD file path", metavar="PATH",
    )
    parser.add_argument(
        "H5_input", type=str, help="HyMD input .H5 file path", metavar="PATH",
    )
    parser.add_argument(
        "--frame", type=str, default=0, metavar="FRAME", nargs="+",
        help=(
            "frame(s) to extract ('all' selects all frames, 'i:j' selects all "
            "frames between i and j)"
        )

    )
    parser.add_argument(
        "--out", type=str, help="output file path", metavar="PATH",
        default=None, required=False,
    )
    parser.add_argument(
        "--scale", type=float, help="scale positions by X", metavar="X",
        default=None, required=False,
    )
    parser.add_argument(
        "--shift", type=float, default=None, required=False, nargs=3,
        help="shift all positions by vector X Y Z",
    )
    args = parser.parse_args()

    make_whole(
        args.H5MD_file, args.H5_input, out_path=args.out, frames=args.frame,
        scale=args.scale, shift=args.shift,
    )
