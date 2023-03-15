import os
import argparse
import h5py
import math
import numpy as np
from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0) # uncomment to suppress warnings


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    From https://stackoverflow.com/a/11541495
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def a2nm(x):
    return x/10.


def genbox(input_file, box_size, prng, charges=None, out_path=None, overwrite=False):
    if out_path is None:
        out_path = os.path.join(os.path.abspath(os.path.dirname(input_file)),
                                os.path.splitext(input_file)[0]+".hdf5")
    if os.path.exists(out_path) and not overwrite:
        error_str = (f'The specified output file {out_path} already exists. '
                     f'use overwrite=True ("-f" flag) to overwrite.')
        raise FileExistsError(error_str)

    # read particles
    particles = {}
    lbl_to_type = {}
    n_particles = 0
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            name, num = line.split()
            lbl_to_type[name] = i
            particles[name] = int(num)
            n_particles += int(num)

    # read charges
    if charges:
        charge_dict = {}
        with open(charges, "r") as f:
            for line in f:
                beadtype, beadchrg = line.split()
                charge_dict[beadtype] = float(beadchrg)

    # write the topology
    with h5py.File(out_path, "w") as out_file:
        position_dataset = out_file.create_dataset(
            "coordinates",
            (1, n_particles, 3,),
            dtype="float32",
        )
        types_dataset = out_file.create_dataset(
            "types",
            (n_particles,),
            dtype="i",
        )
        names_dataset = out_file.create_dataset(
            "names",
            (n_particles,),
            dtype="S10",
        )
        indices_dataset = out_file.create_dataset(
            "indices",
            (n_particles,),
            dtype="i",
        )
        molecules_dataset = out_file.create_dataset(
            "molecules",
            (n_particles,),
            dtype="i",
        )
        bonds_dataset = out_file.create_dataset(
            "bonds",
            (n_particles, 4,),
            dtype="i",
        )
        if charges:
            charges_dataset = out_file.create_dataset(
            "charge",
            (n_particles,),
            dtype="float32",
            )

        bonds_dataset[...] = -1

        # generate the positions randomly inside the box
        initial = 0
        for name, num_beads in particles.items():
            for i in range(initial, initial+num_beads):
                indices_dataset[i] = i
                position_dataset[0, i, :] = np.array(
                                            [prng.uniform(0.,box_size[0]),
                                             prng.uniform(0.,box_size[1]),
                                             prng.uniform(0.,box_size[2])],
                                            dtype=np.float32
                                            )
                charges_dataset[i] = charge_dict[name]
                molecules_dataset[i] = i
                names_dataset[i] = np.string_(name)
                types_dataset[i] = lbl_to_type[name]
            initial += num_beads


if __name__ == '__main__':
    description = 'Generate a hdf5 input placing beads in random positions'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input_file', type=extant_file, help='input .txt with "NAME #particles" for each type')
    parser.add_argument('box_size', type=float, nargs="+", help='input .txt with "NAME #particles" for each type')    
    parser.add_argument('--charges', type=extant_file, default=None,
                        help='file containing the charge for each atom type')
    parser.add_argument('--out', type=str, default=None, dest='out_path',
                        metavar='file name', help='output hymd HDF5 file name')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('-f', action='store_true', default=False, dest='force',
                        help='overwrite existing output file')
    args = parser.parse_args()

    if args.seed is not None:
        ss = np.random.SeedSequence(args.seed)
    else:
        ss = np.random.SeedSequence()
    prng = np.random.default_rng(ss.generate_state(1))

    genbox(args.input_file, args.box_size, prng, charges=args.charges, out_path=args.out_path, overwrite=args.force)