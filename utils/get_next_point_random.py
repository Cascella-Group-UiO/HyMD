import os
import numpy as np
import argparse
from read_parameter_file import read_bounds_file


def get_next_point(bounds_file_path, verbose=True):
    parameter_names_correct_order, lower_bounds, upper_bounds = (
        read_bounds_file(bounds_file_path)
    )
    values = [None for _ in parameter_names_correct_order]
    if verbose:
        print("Picking random points for parameters:")
        print("   |------------------------------------------------------")
    for i, p in enumerate(parameter_names_correct_order):
        val = np.random.uniform(low=lower_bounds[i], high=upper_bounds[i])
        values[i] = val
        if verbose:
            print(f"{i+1:>3}| {p:>5}  {val:>20.15f}  "
                  f"[{lower_bounds[i]:>6.2f}, {upper_bounds[i]:>6.2f}]")
    if verbose:
        print("   |------------------------------------------------------")

    with open(next_point_file_path, "w") as out_file:
        for i, param in enumerate(parameter_names_correct_order):
            out_file.write(f"{values[i]:20.15f}  ")


if __name__ == "__main__":
    description = "Get a random point in parameter space"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=50
        ),
    )
    parser.add_argument(
        "--random-seed", "-rseed", "-seed", dest="seed", type=int,
        default=None, help="seed for the random number generator",
    )
    parser.add_argument(
        "--bounds-file", type=str, default="bounds.txt", metavar="FILE_NAME",
        help="input bounds file path (default 'bounds.txt')",
    )
    parser.add_argument(
        "--out", type=str, default="next_point.txt", metavar="FILE_NAME",
        help="output file path (default 'next_point.txt')",
    )
    parser.add_argument(
        "-s", "--silent", action="store_true", default=False,
        help="suppress terminal output",
    )
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    bounds_file_path = os.path.abspath(args.bounds_file)
    next_point_file_path = os.path.abspath(args.out)

    get_next_point(bounds_file_path, verbose=not args.silent)
