import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description=('Extracts observables from H5MD files and plots them '
                     'using matplotlib. Supported properties: Total energy '
                     '(E), potential energy (PE), kinetic energy (KE), bond '
                     'energy (BE), angular bond energy (AE), and total center '
                     'of mass momentum (P).')
    )
    parser.add_argument('property', default='E', type=str, nargs='+',
                        help='Properties to plot',
                        choices=['E', 'PE', 'KE', 'BE', 'FE', 'AE'])
    parser.add_argument('--file', type=str, default='sim.h5',
                        help='H5MD file')
    parser.add_argument('--per-particle', default=False, action='store_true',
                        help='Divide properties by number of particles')
    parser.add_argument('--subtract-mean', default=False, action='store_true',
                        help=('Subtract the mean, plotting deviations from '
                              'the average'))
    args = parser.parse_args()
    return args


def open_h5md_file(file_path):
    in_file = h5py.File(file_path, 'r')
    return in_file


def close_h5md_file(in_file):
    in_file.close()


def extract_property(h5md_file, property, args):
    observables_group = h5md_file['/observables/']
    particles_group = h5md_file['/particles/all/']
    keyword_to_group_name = {
        'E': 'total_energy', 'PE': 'potential_energy', 'KE': 'kinetic_energy',
        'BE': 'bond_energy', 'AE': 'angle_energy', 'P': 'total_momentum'
    }
    property_group = observables_group[keyword_to_group_name[property]]
    values = property_group['value'][:]
    times = property_group['time'][:]
    plot_kwargs = {'label': keyword_to_group_name[property]}
    xlabel = 'time [ps]'
    ylabel = 'energy [kJ/mol]'
    if args.per_particle:
        n_particles = len(particles_group['position/value'][0, :, 0])
        values /= float(n_particles)
        if args.subtract_mean:
            ylabel = 'energy deviations per particle [kJ/mol]'
        else:
            ylabel = 'energy per particle [kJ/mol]'
    if args.subtract_mean:
        values -= np.mean(values)
    return (times, values, keyword_to_group_name[property], plot_kwargs,
            xlabel, ylabel)


class Property:
    def __init__(self, name, times, values, plot_args, per_particle=False,
                 subtract_mean=False):
        self.name = name
        self.times = times
        self.values = values
        self.plot_args = plot_args
        self.per_particle = per_particle
        self.subtract_mean = subtract_mean


if __name__ == '__main__':
    args = parse_args()
    file_path = os.path.abspath(args.file)
    h5md_file = open_h5md_file(file_path)

    properties = []
    xlabel = None
    ylabel = None
    for p in args.property:
        t, v, n, plot_kwargs, xlabel, ylabel = extract_property(h5md_file, p,
                                                                args)
        properties.append(Property(n, t, v, plot_kwargs,
                                   per_particle=args.per_particle,
                                   subtract_mean=args.subtract_mean))
    close_h5md_file(h5md_file)

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for p in properties:
        ax.plot(p.times, p.values, **p.plot_args)
    ax.legend()
    plt.tight_layout()
    plt.show()
