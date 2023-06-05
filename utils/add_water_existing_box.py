import os
import argparse
import h5py
import numpy as np

def expand_box(input_hdf5, ext_x, ext_y, ext_z, volw, box, out_path, overwrite, prng):
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

    # get the coordinates
    coord = f_in["coordinates"][0,:,:]
    n_particles = coord.shape[0]

    # check box
    if "box" in f_in.attrs and box:
        error_str = "box specified both in the file and command line."
        raise AssertionError(error_str)
    elif not box and "box" in f_in.attrs:
        box = f_in.attrs["box"]
    elif not box and not "box" in f_in.attrs:
        error_str = (
            f"box dimensions not found in the input file ({input_hdf5})"
            f" specify the box size with --original-box"
            )
        raise AssertionError(error_str)
    print(f"Input box size: [{box[0]} {box[1]} {box[2]}]")

    added_particles = 0

    # add dimension in x
    if ext_x != 0.0:
        add_vol = ext_x * box[1] * box[2]
        add_mol = int(np.ceil(add_vol / volw))
        n_particles += add_mol
        added_particles += add_mol

        # randomly insert add_mol beads to f_out
        for _ in range(add_mol):
            new_bead = np.array(
                                [prng.uniform(box[0],box[0]+ext_x),
                                 prng.uniform(0.,box[1]),
                                 prng.uniform(0.,box[2])],
                                dtype=np.float32
                                )
            coord = np.vstack((coord,new_bead))

        # update box
        box[0] += ext_x

    # add dimension in y
    if ext_y != 0.0:
        add_vol = ext_y * box[0] * box[2]
        add_mol = int(np.ceil(add_vol / volw))
        n_particles += add_mol
        added_particles += add_mol

        # randomly insert add_mol beads to f_out
        for _ in range(add_mol):
            new_bead = np.array(
                                [prng.uniform(0.,box[0]),
                                 prng.uniform(box[1],box[1]+ext_y),
                                 prng.uniform(0.,box[2])],
                                dtype=np.float32
                                )
            coord = np.vstack((coord,new_bead))

        # update box
        box[1] += ext_y

    # add dimension in z
    if ext_z != 0.0:
        add_vol = ext_z * box[0] * box[1]
        add_mol = int(np.ceil(add_vol / volw))
        n_particles += add_mol
        added_particles += add_mol

        # randomly insert add_mol beads to f_out
        for _ in range(add_mol):
            new_bead = np.array(
                                [prng.uniform(0.,box[0]),
                                 prng.uniform(0.,box[1]),
                                 prng.uniform(box[2],box[2]+ext_z)],
                                dtype=np.float32
                                )
            coord = np.vstack((coord,new_bead))

        # update box
        box[2] += ext_z

    print(f"{added_particles} will be added to the box (to a total of {n_particles})")

    # find the type of W from the names
    idxw = np.argwhere(f_in["names"][:]==np.string_("W"))[0]
    typew = f_in["types"][idxw]

    # now extend all the other sets
    names = f_in["names"][:]
    for _ in range(added_particles):
        names = np.hstack((names, np.array([np.string_("W")])))

    types = f_in["types"][:]
    for _ in range(added_particles):
        types = np.hstack((types, np.array(typew)))

    last_index = f_in["indices"][-1]
    indices = f_in["indices"][:]
    for i in range(last_index+1,last_index+added_particles+1):
        indices = np.hstack((indices, np.array([i])))

    last_mol = f_in["molecules"][-1]
    molecules = f_in["molecules"][:]
    for i in range(last_mol+1,last_mol+added_particles+1):
        molecules = np.hstack((molecules, np.array([i])))

    if "bonds" in f_in.keys():
        bonds = f_in["bonds"][:]
        for _ in range(added_particles):
            bonds = np.vstack((bonds, np.full((1,4),-1)))
    
        bonds_dataset = f_out.create_dataset(
            "bonds",
            (n_particles, 4,),
            dtype="i",
        )
        bonds_dataset[:] = bonds

    if "charge" in f_in.keys():
        charges = f_in["charge"][:]
        for _ in range(added_particles):
            charges = np.hstack((charges, np.array([0.0])))

        charges_dataset = f_out.create_dataset(
        "charge",
        (n_particles,),
        dtype="float32",
        )
        charges_dataset[:] = charges

    if "velocities" in f_in.keys():
        velocities = f_in["velocities"][0,:,:]
        for _ in range(added_particles):
            velocities = np.vstack((velocities, np.zeros((1,3))))

        velocities_dataset = f_out.create_dataset(
        "velocities",
        (1, n_particles, 3,),
        dtype="float32",
        )
        velocities_dataset[0,:,:] = velocities

    # write to output
    print(f"Output box size: [{box[0]} {box[1]} {box[2]}]")
    f_out.attrs["box"] = box
    f_out.attrs["n_molecules"] = last_mol + added_particles + 1

    position_dataset = f_out.create_dataset(
        "coordinates",
        (1, n_particles, 3,),
        dtype="float32",
    )
    types_dataset = f_out.create_dataset(
        "types",
        (n_particles,),
        dtype="i",
    )
    names_dataset = f_out.create_dataset(
        "names",
        (n_particles,),
        dtype="S10",
    )
    indices_dataset = f_out.create_dataset(
        "indices",
        (n_particles,),
        dtype="i",
    )
    molecules_dataset = f_out.create_dataset(
        "molecules",
        (n_particles,),
        dtype="i",
    )

    position_dataset[0,:,:] = coord
    types_dataset[:] = types
    names_dataset[:] = names
    indices_dataset[:] = indices
    molecules_dataset[:] = molecules

    print(f"Output written to {out_path}.")

    f_in.close()
    f_out.close()


if __name__ == "__main__":
    description = (
        "Extend the box in one or more directions and fill with water (W) beads."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("hdf5_file", type=str, help="input .hdf5 file name")
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
        "--extend-x",
        type=float,
        default=0.0,
        dest="ext_x",
        help="size to expand in x in nm (default 0.0)",
    )
    parser.add_argument(
        "--extend-y",
        type=float,
        default=0.0,
        dest="ext_y",
        help="size to expand in y in nm (default 0.0)",
    )
    parser.add_argument(
        "--extend-z",
        type=float,
        default=0.0,
        dest="ext_z",
        help="size to expand in z in nm (default 0.0)",
    )
    parser.add_argument(
        "--volume-water-bead",
        type=float,
        default=0.1207,
        dest="vol_wat",
        help="volume occupied by a single water bead in nm3 (default 0.1207)",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="random seed"
    )
    parser.add_argument(
        "--original-box",
        type=float,
        nargs="+",
        dest="box",
        help="X, Y and Z dimensions of original box (use if the values are not in the .hdf5)",
    )
    args = parser.parse_args()

    if args.ext_x == 0.0 and args.ext_y == 0.0 and args.ext_z == 0.0:
        err_msg = (
            "At least one of the dimensions of the box should be expanded."
            " Use --extend-x, --extend-y or --extend-z to set the value."
        )
        raise AssertionError(err_msg)

    if args.box:
        if len(args.box) != 3:
            err_msg = "Box receives just 3 values (X, Y and Z)."
            raise AssertionError(err_msg)

    if args.seed is not None:
        ss = np.random.SeedSequence(args.seed)
    else:
        ss = np.random.SeedSequence()
    prng = np.random.default_rng(ss.generate_state(1))

    expand_box(
        args.hdf5_file,
        args.ext_x,
        args.ext_y,
        args.ext_z,
        args.vol_wat,
        args.box,
        args.out_path,
        args.force,
        prng,
    )
