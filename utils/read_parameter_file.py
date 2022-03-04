import os
import argparse


def is_float(string):
    try:
        float(string)
    except (ValueError, TypeError):
        return False
    return True


def read_bounds_file(file_path, verbose=False):
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        raise FileNotFoundError(f"Could not find file {file_path}.")

    with open(file_path, 'r') as in_file:
        contents = in_file.readlines()
        if verbose:
            print(f"Reading file '{file_path}' in directory '{os.getcwd()}':")
            print("   |------------------------------------------------------")
            for i, line in enumerate(contents):
                print(f"{i+1:>3}|", line.rstrip('\n'))
            print("   |------------------------------------------------------")
    parameters, lower_bounds, upper_bounds = [], [], []

    for i, line in enumerate(contents):
        line = line.rstrip('\n')
        if not line.isspace() and not len(line) == 0:
            sline = line.split()
            if len(sline) != 3:
                raise SyntaxError(f"Format of line nr. {i+1} '{line}' of file "
                                  f"'{file_path}' in directory '{os.getcwd()}'"
                                  f" not understood. Expected format 'PARAM "
                                  f"LOWER_BOUND UPPER_BOUND'.")
            if (not is_float(sline[1])) or (not is_float(sline[2])):
                raise ValueError(f"Lower or upper bound '{sline[1]}' or "
                                 f"'{sline[2]}' could not be parsed as a "
                                 f"number.")
            if not len(sline[0].split(',')) == 2:
                raise ValueError(f"Expected two comma separated strings (no "
                                 f"spaces) for parameter name, not "
                                 f"'{sline[0]}' in line nr. {i+1}.")

            parameters.append(sline[0])
            lower_bounds.append(float(sline[1]))
            upper_bounds.append(float(sline[2]))
    return parameters, lower_bounds, upper_bounds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Read and parse Bayesian optimization bounds file"
        )
    )
    parser.add_argument(
        "--bounds-file", type=str, default="bounds.txt", metavar="file name",
        help="input file path (default 'bounds.txt')",
    )
    parser.add_argument(
        "--out", type=str, default="parameters.txt", metavar="file name",
        help="output file path (default 'parameters.txt')",
    )
    parser.add_argument(
        "--silent", action="store_true", default=False,
        help="dont print any output",
    )
    args = parser.parse_args()
    file_path = os.path.abspath(args.bounds_file)
    out_file_path = os.path.abspath(args.out)

    parameters, lower_bounds, upper_bounds = read_bounds_file(
        file_path, verbose=not args.silent,
    )

    with open(out_file_path, 'w') as out_file:
        for parameter in parameters:
            out_file.write(f"{parameter} ")

    with open(out_file_path, 'r') as in_file:
        contents = in_file.readlines()

    if not args.silent:
        print(f"Writing file '{out_file_path}' in directory '{os.getcwd()}':")
        print("   |------------------------------------------------------")
        for i, line in enumerate(contents):
            print(f"{i+1:>3}|", line.rstrip('\n'))
        print("   |------------------------------------------------------")
