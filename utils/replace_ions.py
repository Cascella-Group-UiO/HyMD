import os
import argparse
import h5py
import numpy as np

def replace_ions(input_hdf5, ion_name_in, ion_name_out, ion_charge, out_path, overwrite, prng):
    if out_path is None:
        out_path = os.path.join(
            os.path.abspath(os.path.dirname(input_hdf5)),
            os.path.splitext(os.path.split(input_hdf5)[-1])[0]
            + "_new"
            + os.path.splitext(os.path.split(input_hdf5)[-1])[1],
        )
    if os.path.exists(out_path) and not overwrite:
        error_str = (
            f"The specified output file {out_path} already exists. "
            f'use overwrite=True ("-f" flag) to overwrite.'
        )
        raise FileExistsError(error_str)

    f_in = h5py.File(input_hdf5, "r")
    f_out = h5py.File(out_path, "w")

    # copy from input to output
    for k, v in f_in.attrs.items():
        f_out.attrs[k] = v

    for k in f_in.keys():
        f_in.copy(k, f_out)

    # find ions
    idx_ions = np.argwhere(f_in["names"][:]==np.string_(ion_name_in))
    type_ion = f_in["types"][idx_ions[0]]

    # ion charge
    charge_in = f_in["charge"][idx_ions[0]][0]

    print(f"Total charge for the {idx_ions.shape[0]} ions in the input is {idx_ions.shape[0]*charge_in}")

    # new number of ions
    new_num_ions = round(idx_ions.shape[0]*charge_in / ion_charge - idx_ions.shape[0])
    print(f"Will replace {new_num_ions} water beads by ions")

    # find waters and randomly select some to replace
    idx_w = np.argwhere(f_in["names"][:]==np.string_("W"))
    full_range = np.arange(idx_w.shape[0])
    prng.shuffle(full_range)
    replaced_w_idx = idx_w[np.sort(full_range[:new_num_ions])]

    # print(idx_w)
    # print(replaced_w_idx)

    # replace ion charges and random water charges
    updated_charges = f_in["charge"][...]
    updated_charges[idx_ions] = ion_charge
    updated_charges[replaced_w_idx] = ion_charge

    updated_types = f_in["types"][...]
    updated_types[replaced_w_idx] = type_ion

    updated_names = f_in["names"][...]
    updated_names[idx_ions] = np.string_(ion_name_out)
    updated_names[replaced_w_idx] = np.string_(ion_name_out)

    f_out["charge"][...] = updated_charges
    f_out["types"][...] = updated_types
    f_out["names"][...] = updated_names

    f_in.close()
    f_out.close()

if __name__ == "__main__":
    description = (
        "Replace ions in input hdf5 keeping the box neutral."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("hdf5_file", type=str, help="input .hdf5 file name")
    parser.add_argument(
        "--input-ion",
        type=str,
        default="N",
        help="input ion name (default: N)",
    )
    parser.add_argument(
        "--output-ion",
        type=str,
        default="N",
        help="output ion name (default: N)",
    )
    parser.add_argument(
        "--output-ion-charge",
        type=float,
        default=None,
        help="output ion charge",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        dest="out_path",
        metavar="file name",
        help="output HyMD HDF5 file name",
    )
    parser.add_argument(
        "-f",
        action="store_true",
        default=False,
        dest="force",
        help="overwrite existing output file",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="random seed"
    )
    args = parser.parse_args()

    if args.output_ion_charge is None:
        raise AssertionError("You should provide --output-ion-charge")

    if args.seed is not None:
        ss = np.random.SeedSequence(args.seed)
    else:
        ss = np.random.SeedSequence()
    prng = np.random.default_rng(ss.generate_state(1))

    replace_ions(
        args.hdf5_file,
        args.input_ion,
        args.output_ion,
        args.output_ion_charge,
        args.out_path,
        args.force,
        prng
    )