import numpy as np
# Velocity Verlet
#def integrate_velocity(velocities, accelerations, time_step): #xinmeng
#    #print(masses.reshape(masses.shape[0], 1)) # note this is the mass ratio.. 
#    return velocities + 0.5 * time_step * accelerations #/ masses.reshape(masses.shape[0], 1)

def integrate_velocity(velocities, accelerations, time_step, names, config): #xinmeng
    if not config.freez_types:
        accelerations = accelerations
        
    else:
        for i, group in enumerate(config.freez_types):
            ind = np.where(
                np.logical_or.reduce(list(names == np.string_(t) for t in group))
            )
            accelerations[ind] = accelerations[ind]*0.0 
    
    return velocities + 0.5 * time_step * accelerations #/ masses.reshape(masses.shape[0], 1)

def integrate_position(positions, velocities, time_step):
    return positions + time_step * velocities


# Leap frog?
