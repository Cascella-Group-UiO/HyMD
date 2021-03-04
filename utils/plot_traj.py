import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extracts observables from H5MD files and plots them "
            "using matplotlib. Supported properties: Total energy "
            "(E), potential energy (PE), kinetic energy (KE), bond "
            "energy (BE), angular bond energy (AE), and total center "
            "of mass momentum (P). Plot all available observables by "
            'specifying "all".'
        )
    )
    parser.add_argument(
        "property",
        default="E",
        type=str,
        nargs="+",
        help="Properties to plot",
        choices=["E", "PE", "KE", "BE", "FE", "AE", "P", "all"],
    )  # noqa: E501
    parser.add_argument("--file", type=str, default="sim.h5", help="H5MD file")
    parser.add_argument(
        "--per-particle",
        default=False,
        action="store_true",
        help="Divide properties by number of particles",
    )
    parser.add_argument(
        "--abs", default=False, action="store_true", help="Plot the absolute values"
    )
    parser.add_argument(
        "--subtract-mean",
        default=False,
        action="store_true",
        help=("Subtract the mean, plotting deviations from " "the average"),
    )
    parser.add_argument(
        "--no-plot", default=False, action="store_true", help="Do not show the plot"
    )
    args = parser.parse_args()
    return args


def open_h5md_file(file_path):
    in_file = h5py.File(file_path, "r")
    return in_file


def close_h5md_file(in_file):
    in_file.close()


def print_statistics(name, time, values, n_particles):
    len_name = len(name)
    len_pad = (80 - len_name) // 2 - 2
    title = (
        f'{len_pad * "="} {name.replace("_", " ").capitalize()} '
        f'{len_pad * "="}{"=" if len_name % 2 != 0 else ""}'
    )
    info = f" * {values.shape[0]:>10} samples \n" f" * {time[-1]:>10.3f} ps sampled"
    print(title)
    print(info)

    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    print(f' - {"mean(v):":<35} ', end="")
    for m in mean:
        print(f"{m:>30.15f}", end="")
    print(f'\n - {"std.deviation(v):":<35} ', end="")
    for s in std:
        print(f"{s:>30.15f}", end="")

    std_subs = np.std(values - np.mean(values, axis=0), axis=0)
    print(f'\n - {"std.deviation(v - mean(v)):":<35} ', end="")
    for s in std_subs:
        print(f"{s:>30.15f}", end="")

    mean_per = np.mean(values / n_particles, axis=0)
    std_per = np.std(values / n_particles, axis=0)

    print(f'\n - {"mean(v / N):":<35} ', end="")
    for m in mean_per:
        print(f"{m:>30.15f}", end="")
    print(f'\n - {"std.deviation(v / N):":<35} ', end="")
    for s in std_per:
        print(f"{s:>30.15f}", end="")

    std_per_subs = np.std(
        (values / n_particles) - np.mean(values / n_particles, axis=0), axis=0
    )
    print(f'\n - {"std.deviation(v / N - mean(v / N)):":<35} ', end="")
    for s in std_per_subs:
        print(f"{s:>30.15f}", end="")
    print()


def extract_property(h5md_file, property, args):
    observables_group = h5md_file["/observables/"]
    particles_group = h5md_file["/particles/all/"]
    keyword_to_group_name = {
        "E": "total_energy",
        "PE": "potential_energy",
        "KE": "kinetic_energy",
        "BE": "bond_energy",
        "AE": "angle_energy",
        "FE": "field_energy",
        "P": "total_momentum",
    }
    name = keyword_to_group_name[property]
    property_group = observables_group[name]
    values = property_group["value"][:]
    times = property_group["time"][:]
    plot_kwargs = {"label": keyword_to_group_name[property]}
    xlabel = "time [ps]"
    ylabel = "energy [kJ/mol]"
    n_particles = len(particles_group["position/value"][0, :, 0])

    if args.per_particle:
        values /= float(n_particles)
        if args.subtract_mean:
            ylabel = "energy deviations per particle [kJ/mol]"
        else:
            ylabel = "energy per particle [kJ/mol]"
    if args.subtract_mean:
        values -= np.mean(values)
        ylabel = "abs " + ylabel
    if args.abs:
        values = np.abs(values)
    print_statistics(name, times, property_group["value"][:], n_particles)
    return times, values, name, plot_kwargs, xlabel, ylabel


class Property:
    def __init__(
        self,
        name,
        times,
        values,
        plot_args,
        per_particle=False,
        subtract_mean=False,
        abs=False,
    ):
        self.name = name
        self.times = times
        self.values = values
        self.plot_args = plot_args
        self.per_particle = per_particle
        self.subtract_mean = subtract_mean
        self.abs = abs


if __name__ == "__main__":
    args = parse_args()
    if "all" in args.property:
        args.property = ["E", "PE", "KE", "BE", "FE", "AE", "P"]
    file_path = os.path.abspath(args.file)
    h5md_file = open_h5md_file(file_path)

    properties = []
    xlabel = None
    ylabel = None
    for p in args.property:
        t, v, n, plot_kwargs, xlabel, ylabel = extract_property(h5md_file, p, args)
        properties.append(
            Property(
                n,
                t,
                v,
                plot_kwargs,
                per_particle=args.per_particle,
                subtract_mean=args.subtract_mean,
                abs=args.abs,
            )
        )
    close_h5md_file(h5md_file)

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for p in properties:
        ax.plot(p.times, p.values, **p.plot_args)
    ax.legend()
    plt.tight_layout()
    if not args.no_plot:
        plt.show()
