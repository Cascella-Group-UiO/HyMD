# Velocity Verlet
def integrate_velocity(velocities, accelerations, time_step): #xinmeng
    #print(masses.reshape(masses.shape[0], 1)) # note this is the mass ratio.. 
    return velocities + 0.5 * time_step * accelerations #/ masses.reshape(masses.shape[0], 1)

def integrate_position(positions, velocities, time_step):
    return positions + time_step * velocities


# Leap frog?
