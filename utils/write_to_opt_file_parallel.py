import os
import sys
import argparse
import warnings
import numpy as np


def write_to_opt_file(
    fitness_file_paths,
    parameters_file_path,
    point_file_path,
    opt_file_path
):
    with open(parameters_file_path, "r") as parameters_file:
        parameters = parameters_file.readline().split()
    with open(point_file_path, "r") as point_file:
        point = [float(p) for p in point_file.readline().split()]

    print("Writing fitness data from files:")
    print("   |------------------------------------------------------")
    for i, f in enumerate(fitness_file_paths):
        print(f"{i+1:>3}| {os.path.abspath(f):>5} ")
    print("   |------------------------------------------------------")

    total_fitnesses = []
    for f in fitness_file_paths:
        if os.path.exists(f):
            with open(f, "r") as fitness_file:
                total_fitness = fitness_file.readlines()[3:-1][-1]
                total_fitness = [float(f_) for f_ in total_fitness.split()[1:]]
                total_fitnesses.append(total_fitness)

    N = len(total_fitnesses)  # number of files
    if N == 0:
        warn_str = (
            f"No fitness data found in paths {fitness_file_paths}"
        )
        warnings.warn(warn_str)
        sys.exit(0)
    M = len(total_fitnesses[0])  # number of metrics
    total_fitnesses_array = np.empty(shape=(N, M,), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            total_fitnesses_array[i, j] = total_fitnesses[i][j]

    new_file = False
    if not os.path.exists(opt_file_path):
        new_file = True
        open_key = "w"
    else:
        open_key = "a"

    with open(opt_file_path, open_key) as out_file:
        print(f"Writing to opt. data file: {os.path.abspath(opt_file_path)}")
        if new_file:
            for parameter in parameters:
                out_file.write(f"{parameter:>15}")
            for f in ("MSE", "RMSE", "MAE", "MAPE", "R2", "SMAPE"):
                out_file.write(f"{f:>25}")
            out_file.write("\n")

        for p in point:
            out_file.write(f"{p:>15.10f}")
        for i in range(M):
            f = np.mean(total_fitnesses_array[:, i])
            out_file.write(f"{f:>25.15g}")
        out_file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Write data from multiple fitness files to the collective opt-file"
        )
    )
    parser.add_argument(
        "--fitness-files", type=str, default=None, metavar="file names",
        nargs="+", dest="fitness_files", required=True,
        help="multiple input fitness file paths",
    )
    parser.add_argument(
        "--parameters-file", type=str, default=None, metavar="file name",
        dest="parameters_file", help="parameters file path", required=True,
    )
    parser.add_argument(
        "--point-file", type=str, default=None, metavar="file name",
        dest="point_file", help="next point file path", required=True,
    )
    parser.add_argument(
        "--opt-file", type=str, default="opt_data.txt", metavar="file name",
        dest="opt_file", help="output opt data file (default opt_data.txt)",
    )
    args = parser.parse_args()
    write_to_opt_file(
        args.fitness_files,
        args.parameters_file,
        args.point_file,
        args.opt_file
    )
