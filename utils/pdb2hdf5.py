import os
import argparse
import h5py
import math
import numpy as np
from openbabel import openbabel

openbabel.obErrorLog.SetOutputLevel(0)  # uncomment to suppress warnings


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
    return x / 10.0


def pdb_to_input(
    pdb_file, box=None, charges=None, bonds=None, out_path=None, overwrite=False
):
    if out_path is None:
        out_path = os.path.join(
            os.path.abspath(os.path.dirname(pdb_file)),
            os.path.splitext(pdb_file)[0] + ".hdf5",
        )
    if os.path.exists(out_path) and not overwrite:
        error_str = (
            f"The specified output file {out_path} already exists. "
            f'use overwrite=True ("-f" flag) to overwrite.'
        )
        raise FileExistsError(error_str)

    # read charges
    if charges:
        charge_dict = {}
        with open(charges, "r") as f:
            for line in f:
                beadtype, beadchrg = line.split()
                charge_dict[beadtype] = float(beadchrg)

    # read bonds
    bonds_list = []
    if bonds:
        with open(bonds, "r") as f:
            for line in f:
                bnd1 = "{} {}".format(line.split()[0], line.split()[1])
                bnd2 = "{} {}".format(line.split()[1], line.split()[0])
                bonds_list.append(bnd1)
                bonds_list.append(bnd2)

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "xyz")
    # ignore ring perception and CONECTs for faster processing
    obConversion.AddOption("b", openbabel.OBConversion.INOPTIONS)
    obConversion.AddOption("c", openbabel.OBConversion.INOPTIONS)

    # read molecule to OBMol object
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, pdb_file)
    # if not bonds:
    #     mol.ConnectTheDots() # necessary because of the 'b' INOPTION
    n_particles = mol.NumAtoms()
    print("Found {} particles in .pdb".format(n_particles))

    # get atomic labels from pdb
    atom_to_lbl = {}
    for res in openbabel.OBResidueIter(mol):
        for atom in openbabel.OBResidueAtomIter(res):
            atom_to_lbl[atom.GetId()] = res.GetAtomID(atom).strip()

    # from label to type
    lbl_to_type = {}
    for i, lbl in enumerate(set(atom_to_lbl.values())):
        lbl_to_type[lbl] = i

    # if bonds per residue are passed, connect atoms
    for res in openbabel.OBResidueIter(mol):
        for atom1 in openbabel.OBResidueAtomIter(res):
            for atom2 in openbabel.OBResidueAtomIter(res):
                id1, id2 = atom1.GetId(), atom2.GetId()
                if id1 == id2:
                    continue
                if "{} {}".format(atom_to_lbl[id1], atom_to_lbl[id2]) in bonds_list:
                    mol.AddBond(id1 + 1, id2 + 1, 1)

    molecules = mol.Separate()
    n_molecules = len(molecules)

    # detect the molecules types
    atom_to_mol = {}
    for i, submol in enumerate(molecules):
        for at in openbabel.OBMolAtomIter(submol):
            atom_to_mol[at.GetId()] = i

    # write the topology
    with h5py.File(out_path, "w") as out_file:
        if box:
            out_file.attrs["box"] = np.array(box)
        out_file.attrs["n_molecules"] = n_molecules
        position_dataset = out_file.create_dataset(
            "coordinates",
            (
                1,
                n_particles,
                3,
            ),
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
            (
                n_particles,
                4,
            ),
            dtype="i",
        )
        if charges:
            charges_dataset = out_file.create_dataset(
                "charge",
                (n_particles,),
                dtype="float32",
            )

        bonds_dataset[...] = -1
        for i, atom in enumerate(openbabel.OBMolAtomIter(mol)):
            indices_dataset[i] = i
            position_dataset[0, i, :] = np.array(
                [a2nm(atom.GetX()), a2nm(atom.GetY()), a2nm(atom.GetZ())],
                dtype=np.float32,
            )
            charges_dataset[i] = charge_dict[atom_to_lbl[atom.GetId()]]
            molecules_dataset[i] = atom_to_mol[atom.GetId()]
            names_dataset[i] = np.string_(atom_to_lbl[atom.GetId()])
            types_dataset[i] = lbl_to_type[atom_to_lbl[atom.GetId()]]
            if len(bonds_list) != 0:
                for j, nbr in enumerate(openbabel.OBAtomAtomIter(atom)):
                    bonds_dataset[i, j] = nbr.GetId()


if __name__ == "__main__":
    description = "Convert .pdb files to the hymd hdf5 input format"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("pdb_file", type=extant_file, help="input .pdb file name")
    parser.add_argument(
        "--charges",
        type=extant_file,
        default=None,
        help="file containing the charge for each atom type",
    )
    parser.add_argument(
        "--bonds",
        type=extant_file,
        default=None,
        help="file containing the connected beads for each residue",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        dest="out_path",
        metavar="file name",
        help="output hymd HDF5 file name",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=None,
        nargs="+",
        help="box dimensions",
    )
    parser.add_argument(
        "-f",
        action="store_true",
        default=False,
        dest="force",
        help="overwrite existing output file",
    )
    args = parser.parse_args()

    # get basename and file extension
    base, ext = os.path.splitext(args.pdb_file)

    # check input consistency
    if ext.lower()[1:] != "pdb":
        parser.error("pdb_file extension is not .pdb")

    if args.box:
        if len(args.box) != 3:
            parser.error("should provide 3 dimensional box (3 floats)")

    pdb_to_input(
        args.pdb_file,
        box=args.box,
        charges=args.charges,
        bonds=args.bonds,
        out_path=args.out_path,
        overwrite=args.force,
    )
