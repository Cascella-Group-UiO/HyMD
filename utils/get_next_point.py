import os
import numpy as np
import argparse
from read_parameter_file import read_bounds_file
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


def get_next_point_BO(bounds_file_path, opt_file_path, kappa):
    if os.path.exists(opt_file_path):
        opt_data = np.loadtxt(opt_file_path, dtype=str)
    else:
        with open(opt_file_path, "w") as _:
            ...
        opt_data = np.empty(shape=(1, 1,), dtype=np.float64)

    parameter_names_correct_order, lower_bounds, upper_bounds = (
        read_bounds_file(bounds_file_path)
    )
    bounds = {}
    for param, lower, upper in zip(
        parameter_names_correct_order, lower_bounds, upper_bounds
    ):
        bounds[param] = (lower, upper)

    optimizer = BayesianOptimization(f=None, pbounds=bounds, verbose=2)
    white_kernel = WhiteKernel(noise_level_bounds=(1e-5, 1e+3))
    matern_kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3))
    kernel = matern_kernel + white_kernel
    optimizer._gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=100, alpha=0.0
    )
    utility = UtilityFunction(kind="ucb", kappa=args.kappa, xi=None)

    print("Registering optimization results for previous runs:")
    print(f" {'':4} |------------------------------------------------------")
    print(f" {'':<4} | ", end="")
    for parameter in parameter_names_correct_order:
        print(f"{parameter:>20} ", end="")
    print(f"{args.fitness:>20}")
    print(f" {'':4} |------------------------------------------------------")

    for i in range(opt_data.shape[1]):
        if opt_data[0, i] == args.fitness:
            target_ind = i
            break

    param = []
    target = []

    for iteration in range(1, opt_data.shape[0]):
        param.append({})
        for i in range(len(parameter_names_correct_order)):
            param[-1][opt_data[0, i]] = float(opt_data[iteration, i])
        target.append(float(opt_data[iteration, target_ind]))

    target = np.array(target, dtype=np.float64)
    target_flip = np.array(target, dtype=np.float64)
    max_target = np.max(target)

    for i, t in enumerate(target):
        if args.fitness in ("MSE", "RMSE", "MAE", "MAPE", "SMAPE"):
            target_flip[i] = max_target - t
        else:
            target_flip[i] = t

    iteration = 0
    for p, t, t_print in zip(param, target_flip, target):
        iteration += 1
        added = False
        tries = 0
        random_key = None
        while not added:
            try:
                if tries > 0:
                    random_key = np.random.choice(list(p.keys()), 1)[0]
                    p[random_key] += np.random.normal(loc=0.0, scale=1e-14)
                optimizer.register(params=p, target=t)
                added = True
            except KeyError:
                tries += 1

        print(f" {iteration:>4} | ", end="")
        for parameter in parameter_names_correct_order:
            print(f"{p[parameter]:20.15f} ", end="")
        print(f"{t_print:20.15f} ",
              end=f" ({tries}, {random_key})\n" if tries != 0 else "\n")
    print(f" {'':4} |------------------------------------------------------")

    print("Evaluating acquisition function for next parameter space point:")
    print("   |------------------------------------------------------")
    next_point = optimizer.suggest(utility)
    for i, parameter in enumerate(parameter_names_correct_order):
        print(f"{i+1:>3}| {parameter:>5}  {next_point[parameter]:>20.15f}  "
              f"[{lower_bounds[i]:>6.2f}, {upper_bounds[i]:>6.2f}]")
    print("   |------------------------------------------------------")

    with open(next_point_file_path, "w") as out_file:
        for parameter in parameter_names_correct_order:
            out_file.write(f"{next_point[parameter]:20.15f}")
        out_file.write("")

    print("Kernel hyper-parameters fitted to the data (max. log. likelihood):")
    print("   |------------------------------------------------------")
    print("   | ", optimizer._gp.kernel)
    print("   |------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Get the next point in parameter space, acquisition func maximum"
        )
    )
    parser.add_argument(
        "--random-seed", "-rseed", "-seed", dest="seed", type=int,
        default=None, help="seed for the random number generator"
    )
    parser.add_argument(
        "--bounds-file", type=str, default="bounds.txt", metavar="FILE_NAME",
        help="input bounds file path (default 'bounds.txt')",
    )
    parser.add_argument(
        "--parameters-file", type=str, default="parameters.txt",
        metavar="FILE_NAME",
        help="input bounds file path (default 'parameters.txt')",
    )
    parser.add_argument(
        "--opt-file", type=str, default="opt_data.txt",
        metavar="FILE_NAME",
        help="input bounds file path (default 'opt_data.txt')",
    )
    parser.add_argument(
        "--out", type=str, default="next_point.txt", metavar="FILE_NAME",
        help="output file path (default 'next_point.txt')",
    )
    parser.add_argument(
        "--fitness", type=str, default="R2",
        help=(
            "name of the fitness function to use for the acquisition function"
        ),
    )
    parser.add_argument(
        "--kappa", type=float, default=1.0, dest="kappa",
        help="exploration/exploitation tradeoff parameter",
    )

    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    bounds_file_path = os.path.abspath(args.bounds_file)
    parameters_file_path = os.path.abspath(args.parameters_file)
    next_point_file_path = os.path.abspath(args.out)
    opt_file_path = os.path.abspath(args.opt_file)

    get_next_point_BO(
        bounds_file_path, opt_file_path, args.kappa,
    )
