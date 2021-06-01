import h5py
import numpy as np 
import matplotlib.pyplot as plt 

### h5dump -d '/parameters/vmd_structure/name' sim.h5

def read_hdf5_file_vel(hdf5_file_name):
    in_file = h5py.File(hdf5_file_name, 'r')
    species_to_name_map = {0: 'N', 1: 'P', 2: 'G', 3: 'C', 4: 'W', 5: 'F'}
    species = in_file['particles']['all']['species'][:]
    positions = in_file['particles']['all']['position']['value'][...]
    time = in_file['particles']['all']['position']['time'][...]
    
    if 'velocity' in in_file['particles']['all']:
        velocities = in_file['particles']['all']['velocity']['value'][...]
    else:
        velocity = None
    
    if 'force' in in_file['particles']['all']:
        forces = in_file['particles']['all']['force']['value'][...]
    else:
        forces = None
    in_file.close()
    return positions, velocities, forces, species, species_to_name_map, time

def species_temperature(positions, velocities, forces, species, species_to_name_map, time):
    mass = 72.0
    kB = 0.008314
    
    n_frames = positions.shape[0]
    n_particles = positions.shape[1]
    unique_species = np.array(list(species_to_name_map.keys()), dtype=int)
    n_unique_species = len(unique_species)
    
    inds = []
    for s in unique_species:
        inds.append(np.where(species == s))
    n_species = np.empty(shape=(len(inds,)), dtype=int)
    for i, ind in enumerate(inds):
        n_species[i] = ind[0].size
    
    colors = ['darkblue', 'gold', 'lightcoral', 'teal', 'aqua', 'red']    
    marker = ['-^', '-x', '-v', '-o', '-s', '->']
    fig = plt.figure(figsize=(15,8))
    plotkwargs = {'fontsize': 15}
    
    kinetic_energy_species = np.empty(shape=(n_unique_species, n_frames), dtype=np.float64)
    temperature_species = np.empty_like(kinetic_energy_species, dtype=np.float64)
    total_temperature = np.zeros(shape=(n_frames,), dtype=np.float64)
    for s in unique_species:
        v_squared = np.sum(np.squeeze(velocities[:, inds[s], :]**2), axis=2, keepdims=False)
        v_squared = np.sum(v_squared, axis=1, keepdims=True)
        v_squared = np.squeeze(v_squared)
        kinetic_energy_species[s, :] = 0.5 * mass * v_squared
        temperature_species[s, :] = 2.0 * kinetic_energy_species[s, :] / (kB * 3.0 * n_species[s])
        total_temperature[:] += temperature_species[s, :] * n_species[s]
        plt.plot(time, temperature_species[s, :], marker[s], color=colors[s], label=species_to_name_map[s])

    plt.axhline(323, color='green', label='target T')
    plt.plot(time, total_temperature / n_particles, 'k--', label='total T')
    plt.legend(loc='upper right', **plotkwargs)
    plt.xlabel('time / ps', **plotkwargs)
    plt.ylabel('temperature / K', **plotkwargs)
    plt.show()

file_path = '/Users/mortenledum/Documents/hymd_paper/bilayer_stability/sim_equil_2.h5'
file_path = './sim.h5'
positions, velocities, forces, species, species_to_name_map, time = read_hdf5_file_vel(file_path)
_ = species_temperature(positions, velocities, forces, species, species_to_name_map, time)


