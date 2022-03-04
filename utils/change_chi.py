import warnings
import argparse
import os
import itertools
import re


def parse_and_change_config_chi(config_file_path, parser, out=None):
    """Parse and change the chi-matrix in a config file

    Parameters
    ----------
    config_file_path : str
        File path of the config file to change.
    parser : Argparse.Namespace
        Parsed command line arguments given to change_chi.
    out : str, optional
        File path of the changed output config file.

    Returns
    -------
    out_file_path : str
        File path of the changed output config file. None if no out parameter
        was specified.
    out_file_contents : list of int
        Lines of the changed output config file.
    """
    with open(config_file_path, "r") as in_file:
        contents = in_file.readlines()
    chi_re = re.compile(r"^\s*chi\s*=\s*\[\s*")

    chi_start_line = None
    for i, line in enumerate(contents):
        chi_match = re.match(chi_re, line)
        if chi_match is not None:
            chi_start_line = i
            break

    chi_end_line = None
    left_brackets = 0
    right_brackets = 0
    for i, line in enumerate(contents[chi_start_line:]):
        for j, char in enumerate(line):
            if char == "[":
                left_brackets += 1
            elif char == "]":
                right_brackets += 1
            if left_brackets > 0:
                if left_brackets == right_brackets:
                    chi_end_line = chi_start_line + i
                    break
        if chi_end_line is not None:
            break

    chi_arg_re = re.compile(r"[A-Z,a-z]+,[A-Z,a-z]+")
    chi_args = []
    for arg, value in vars(parser).items():
        if value is not None:
            if re.fullmatch(chi_arg_re, arg):
                a, b = arg.split(",")
                re_str = (
                    f'\[\s*\[\s*"{a}"\s*,\s*"{b}"\s*\]\s*,'  # noqa: W605
                    f'\s*\[\s*[\+-]?\d+\.\d*\s*\]\s*\]'  # noqa: W605
                )
                re_str_r = (
                    f'\[\s*\[\s*"{b}"\s*,\s*"{a}"\s*\]\s*,'  # noqa: W605
                    f'\s*\[\s*[\+-]?\d+\.\d*\s*\]\s*\]'  # noqa: W605
                )
                chi_matrix_re = re.compile(re_str)
                chi_matrix_re_r = re.compile(re_str_r)
                chi_args.append([(a, b), chi_matrix_re, value])
                chi_args.append([(a, b), chi_matrix_re_r, value])

    for i, line in enumerate(contents[chi_start_line:chi_end_line+1]):
        replaced_line = line
        for (a, b), pattern, chi_value in chi_args:
            replaced_line = re.sub(
                pattern,
                f'[["{a}", "{b}"], [{chi_value}]]',
                replaced_line,
            )
        contents[chi_start_line+i] = replaced_line

    if out is not None:
        with open(out, "w") as out_file:
            for line in contents:
                out_file.write(line)
    return contents


if __name__ == '__main__':
    description = ""
    parser = argparse.ArgumentParser(
        description=(
            "Change elements in the chi-matrix in config.toml"
        )
    )
    parser.add_argument(
        "--config-file", type=str, metavar="config file",
        help="config.toml file to edit",
    )
    parser.add_argument(
        "--out", type=str, default=None, metavar="file name",
        help="output file path (config file with changed chi matrix)",
    )
    parser.add_argument(
        "-w", action="store_true", help="suppress all user warnings",
        dest="suppress_warnings",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose",
        help="increase logging verbosity",
    )
    beads = ["N", "P", "G", "C", "D", "L", "W"]
    combinations = itertools.combinations(beads, 2)
    for a, b in combinations:
        bead_1, bead_2 = sorted((a, b,))
        parser.add_argument(
            f"-{bead_1},{bead_2}", f"-{bead_2},{bead_1}", type=float,
            dest=f"{bead_1},{bead_2}", default=None, metavar="",
            help=f"{bead_1}-{bead_2} interaction energy",
        )
    args = parser.parse_args()

    config_file_path = os.path.abspath(args.config_file)
    out_file_path = None
    if args.out is None:
        if not args.suppress_warnings:
            warnings.warn(
                "No output file specified, dumping result to stdout."
            )
    else:
        out_file_path = os.path.abspath(args.out)

    replaced_file_content = parse_and_change_config_chi(
        config_file_path, args, out=out_file_path
    )
    if (args.out is not None and args.verbose) or args.out is None:
        for line in replaced_file_content:
            print(line.strip("\n"))
