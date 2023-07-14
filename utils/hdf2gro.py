import os
import argparse
import h5py
import numpy as np
import tomli


def create_gro(hdf5_file, toml_file, out_path=None, overwrite=False):
    if out_path is None:
        out_path = os.path.join(
            os.path.abspath(os.path.dirname(hdf5_file)),
            os.path.splitext(os.path.split(hdf5_file)[-1])[0] + ".gro",
        )
    if os.path.exists(out_path) and not overwrite:
        error_str = (
            f"The specified output file {out_path} already exists. "
            f'use overwrite=True ("-f" flag) to overwrite.'
        )
        raise FileExistsError(error_str)

    # read toml
    with open(toml_file, "rb") as f:
        topol = tomli.load(f)

    top_mol = topol["system"]["molecules"]

    start = 0
    maxidx_to_name = []
    for name, nmol in top_mol:
        maxidx_to_name.append((start + 1, nmol + start, name))
        start += nmol

    # read hdf5
    f_hd5 = h5py.File(hdf5_file, "r")

    fp = open(out_path, "w")

    shape = f_hd5["coordinates"].shape[0]
    names = f_hd5["names"][:]
    Np = f_hd5["coordinates"].shape[1]
    if "molecules" in f_hd5:
        molecules = f_hd5["molecules"][:]

    for f in range(shape):
        fp.write("MD of %d mols\n" % (Np))
        fp.write("%-10d\n" % (Np))
        for i in range(Np):
            name = names[i].decode("UTF-8").strip()

            resnum = molecules[i] + 1

            resname = None
            for starti, endi, namei in maxidx_to_name:
                if resnum >= starti and resnum <= endi:
                    resname = namei
                    break

            if resname is None:
                raise AssertionError(f"Residue name for molecule {resnum-1} not found")

            dump_str = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n" % (
                resnum % 100000,
                resname,
                name,
                (i + 1) % 100000,
                f_hd5["coordinates"][f, i, 0],
                f_hd5["coordinates"][f, i, 1],
                f_hd5["coordinates"][f, i, 2],
                f_hd5["velocities"][f, i, 0],
                f_hd5["velocities"][f, i, 1],
                f_hd5["velocities"][f, i, 2],
            )
            fp.write(dump_str)
        fp.write(
            "%-5.5f\t%5.5f\t%5.5f\n"
            % (
                f_hd5.attrs["box"][0],
                f_hd5.attrs["box"][1],
                f_hd5.attrs["box"][2],
            )
        )
        fp.flush()

    fp.close()


if __name__ == "__main__":
    description = "Convert .hdf5 file and .toml topology to .gro topology to be used with MDAanalysis"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("hdf5_file", type=str, help="input .hdf5 file name")
    parser.add_argument("toml_file", type=str, help="topology in .toml file")
    parser.add_argument(
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

    create_gro(args.hdf5_file, args.toml_file, args.out_path, args.force)
