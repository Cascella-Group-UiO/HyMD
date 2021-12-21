# Velocity Verlet
def integrate_velocity(velocities, accelerations, time_step):
    return velocities + 0.5 * time_step * accelerations


def integrate_position(positions, velocities, time_step):
    return positions + time_step * velocities
