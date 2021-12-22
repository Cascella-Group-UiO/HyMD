import pytest
import numpy as np
from hymd.integrator import integrate_velocity, integrate_position


def test_integrator_velocity_zero_force(three_atoms):
    _, _, _, _, _, velocities, CONF = three_atoms
    old_velocities = velocities.copy()
    velocities = integrate_velocity(velocities, np.zeros_like(velocities),
                                    CONF['dt'])
    assert np.allclose(old_velocities, velocities, atol=1e-13)


def test_integrator_velocity(three_atoms):
    _, _, _, _, _, velocities, CONF = three_atoms
    accelerations = np.array([
        [5.309962614030253,  -5.068488339109831,  0.828090598458989],
        [-3.868819942226709, -9.720221218265905,  4.036326041075613],
        [9.728114214745183,  -8.189855111379165, -3.075260427419437]
    ], dtype=np.float64)
    velocities = integrate_velocity(velocities, accelerations, CONF['dt'])
    expected_velocities = np.array([
        [0.3004100792669800, -0.452528545222956, -0.288323467370162],
        [-0.153011424114385,  0.313703442307804,  0.420088246650352],
        [0.4402634172106540,  0.077338401336559,  0.041682352532849]
    ], dtype=np.float64)
    assert np.allclose(expected_velocities, velocities, atol=1e-13)


def test_integrator_position_zero_velocity_zero_force(three_atoms):
    _, _, _, _, positions, velocities, CONF = three_atoms
    old_positions = positions.copy()
    positions = integrate_position(positions, np.zeros_like(velocities),
                                   CONF['dt'])
    assert np.allclose(old_positions, positions, atol=1e-13)


def test_integrator_position_zero_force(three_atoms):
    _, _, _, _, positions, velocities, CONF = three_atoms
    positions = integrate_position(positions, velocities, CONF['dt'])
    expected_positions = np.array([
        [3.341937188559520, 3.053303418939576, 3.648148235458336],
        [4.254612887747975, 3.315211337414966, 4.720346322206166],
        [2.170585066308332, 2.215015584448489, 4.184927411402398]
    ], dtype=np.float64)
    assert np.allclose(expected_positions, positions, atol=1e-13)


def test_integrator_position(three_atoms):
    _, _, _, _, positions, velocities, CONF = three_atoms
    positions = integrate_position(positions, velocities, CONF['dt'])
    expected_positions = np.array([
        [3.341937188559520, 3.053303418939576, 3.648148235458336],
        [4.254612887747975, 3.315211337414966, 4.720346322206166],
        [2.170585066308332, 2.215015584448489, 4.184927411402398]
    ], dtype=np.float64)
    assert np.allclose(expected_positions, positions, atol=1e-13)


def test_integrator_velocity_verlet_constant_force(three_atoms):
    """With a constant force F₀ = (α, β, γ), the velocity obeys the recurrence
    relation

        v       =   v      +  Δt  a
          i+1         i             0

    with solution

        v       =   v      +  i Δt a
          i           0              0

    The position obeys the recurrence relation
                                 ╭     ╮       1    2
        p       =   p      +  Δt │ v   │   +  ─── Δt  a
          i+1         i          ╰   i ╯       2        0
                                 ╭                 ╮       1    2
                =   p      +  Δt │ v   +  i Δt a   │   +  ─── Δt  a
                      i          ╰   0           0 ╯       2        0

    with solution
                               i Δt  ╭                      ╮
        p       =   p      +  ────── │ a   Δt i   +   2 v   │
          i           0         2    ╰   0                0 ╯
    """
    _, _, _, _, positions, velocities, CONF = three_atoms
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()
    dt = CONF['dt']
    accelerations = np.array([
        [-7.539285072366672,  0.968464568407796,  0.598639645160741],
        [9.1962508228909070,  1.505356899114361,  3.601585312042335],
        [8.8869056807446400, -0.972763307999720, -7.335044988262844]
    ], dtype=np.float64)

    for step in range(1, 10):
        # Velocity Verlet steps
        velocities = integrate_velocity(velocities, accelerations, dt)
        positions = integrate_position(positions, velocities, dt)
        velocities = integrate_velocity(velocities, accelerations, dt)

        for i in range(3):
            for d in range(3):
                v = velocities[i, d]
                a = accelerations[i, d]
                v_0 = initial_velocities[i, d]
                v_expected = v_0 + step * dt * a
                assert v == pytest.approx(v_expected, abs=1e-13)

                p = positions[i, d]
                p_0 = initial_positions[i, d]
                p_expected = p_0 + 0.5 * step * dt * (a * dt * step + 2 * v_0)
                assert p == pytest.approx(p_expected, abs=1e-13)


def test_integrator_velocity_verlet_linear_force(three_atoms):
    """With a time-linear force F(t) = (α t, β t, γ t) the velocity obeys the
    relation
                             α Δt ╭            ╮
       v       =   v     +  ───── │ 2 t  +  Δt │
         i+1         0        2   ╰            ╯

    with solution
                             1      2   2
       v       =   v     +  ─── α Δt  i               (1)
         i           0       2

    The position recurrence relation takes the form
                               ╭         α    2  2 ╮    1      3
       p       =   p     +  Δt │ v    + ─── Δt  i  │ + ─── α Δt  i
         i+1         0         ╰   0     2         ╯    2

    with solution
                             1      3   ╭  2      ╮
       p       =   p     +  ─── α Δt  i │ i  -  1 │ +  Δt i v
         i           0       6          ╰         ╯           0
    """
    _, _, _, _, positions, velocities, CONF = three_atoms
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()
    dt = CONF['dt']
    acceleration_constants = np.array([
        [-7.539285072366672,  0.968464568407796,  0.598639645160741],
        [9.1962508228909070,  1.505356899114361,  3.601585312042335],
        [8.8869056807446400, -0.972763307999720, -7.335044988262844]
    ], dtype=np.float64)

    # Initial forces at t=0 are zero
    accelerations = np.zeros_like(acceleration_constants)
    for step in range(1, 10):
        # Velocity Verlet steps
        velocities = integrate_velocity(velocities, accelerations, dt)
        positions = integrate_position(positions, velocities, dt)

        # Update forces
        accelerations = step * dt * acceleration_constants
        velocities = integrate_velocity(velocities, accelerations, dt)

        for i in range(3):
            for d in range(3):
                v = velocities[i, d]
                a_0 = acceleration_constants[i, d]
                v_0 = initial_velocities[i, d]
                v_expected = v_0 + 0.5 * a_0 * dt**2 * step**2
                assert v == pytest.approx(v_expected, abs=1e-13)

                p = positions[i, d]
                p_0 = initial_positions[i, d]
                p_expected = (p_0 + (step**2 - 1) * a_0 * dt**3 * step / 6.0 +
                              dt * step * v_0)
                assert p == pytest.approx(p_expected, abs=1e-13)


def test_integrator_respa_velocity_verlet_equivalent(three_atoms):
    _, _, _, _, positions_vv, velocities_vv, CONF = three_atoms
    positions_respa = positions_vv.copy()
    velocities_respa = velocities_vv.copy()
    Dt = CONF['dt']
    M = 1  # Inner RESPA steps per long Δt steps
    dt = Dt / M

    def acceleration_long(positions):
        equilibrium = np.array([
            [5.5643712591568950,  0.884023577514768,  9.170162581620790],
            [2.8788490727172800,  3.400482961264622, -1.520671282680709],
            [-1.487026411226955, -6.859348424240592, -2.549294472000168]
        ], dtype=np.float64)
        return (positions - equilibrium)**2

    def acceleration_short(positions):
        return - positions

    a_l_vv = acceleration_long(positions_vv)
    a_s_vv = acceleration_short(positions_vv)
    a_l_respa = acceleration_long(positions_respa)
    a_s_respa = acceleration_short(positions_respa)

    for step in range(1, 10):
        # RESPA steps
        velocities_respa = integrate_velocity(velocities_respa, a_l_respa, Dt)

        for s in range(M):
            velocities_respa = integrate_velocity(velocities_respa, a_s_respa,
                                                  dt)
            positions_respa = integrate_position(positions_respa,
                                                 velocities_respa, dt)

            a_s_respa = acceleration_short(positions_respa)
            velocities_respa = integrate_velocity(velocities_respa, a_s_respa,
                                                  dt)

        a_l_respa = acceleration_long(positions_respa)
        velocities_respa = integrate_velocity(velocities_respa, a_l_respa, Dt)

        # Velocity Verlet steps
        velocities_vv = integrate_velocity(velocities_vv, a_l_vv + a_s_vv, Dt)
        positions_vv = integrate_position(positions_vv, velocities_vv, Dt)
        a_l_vv = acceleration_long(positions_vv)
        a_s_vv = acceleration_short(positions_vv)
        velocities_vv = integrate_velocity(velocities_vv, a_l_vv + a_s_vv, Dt)

        assert np.allclose(velocities_vv, velocities_respa, atol=1e-13)
        assert np.allclose(positions_vv, positions_respa, atol=1e-13)


def test_integrator_respa_constant_force(three_atoms):
    """With a constant force F₀ = (α, β, γ), the velocity obeys the recurrence
    relation (dt = Δt / M, with M the number of inner RESPA steps per long Δt
    step)

        v       =   v      +  Δt a      +  M dt a
          i+1         i            L              S

    with solution
                                ╭                       ╮
        v       =   v      +  i │ Δt a      +  M dt a   │
          i           0         ╰      L              S ╯

    The position obeys the recurrence relation                  2
                        M dt  ╭                         ╮     dt  aS (M - 1) M
        p    =  p   +  ────── │ 2 v   + Δt a   + dt a   │ +  ──────────────────
         i+1      i      2    ╰     i        L        S ╯            4
    """
    _, _, _, _, initial_positions, initial_velocities, CONF = three_atoms
    Dt = CONF['dt']
    acceleration_long = np.array([
        [1.6851177297118020,  2.849556517742162,  1.109363734104484],
        [-3.129093587905032,  1.363258438204340,  3.621192799979525],
        [0.2028711865869220, -3.176012574568037, -2.357588753443123]
    ], dtype=np.float64)
    acceleration_short = np.array([
        [-6.162908853722353,  2.993015003002288,  9.462957452628899],
        [2.2878276475508310, -4.867859096429122, -8.123570401997174],
        [-7.826918369186258, -3.199027970628050, -6.067654493641959]
    ], dtype=np.float64)

    for respa in (1, 2, 3, 4, 5, 10, 20):
        M = respa
        dt = Dt / M

        velocities = initial_velocities.copy()
        positions = initial_positions.copy()

        for step in range(1, 10):
            v_o = velocities.copy()
            p_o = positions.copy()

            # RESPA steps
            velocities = integrate_velocity(velocities, acceleration_long, Dt)

            for s in range(M):
                velocities = integrate_velocity(velocities, acceleration_short,
                                                dt)
                positions = integrate_position(positions, velocities, dt)

                # Calculate forces_short here
                velocities = integrate_velocity(velocities, acceleration_short,
                                                dt)

            # Calculate forces_long here
            velocities = integrate_velocity(velocities, acceleration_long, Dt)

            for i in range(3):
                for d in range(3):
                    v = velocities[i, d]
                    v_0 = initial_velocities[i, d]
                    p = positions[i, d]
                    # p_0 = initial_positions[i, d]
                    a_s = acceleration_short[i, d]
                    a_l = acceleration_long[i, d]
                    v_expected = v_0 + step * (Dt * a_l + M * dt * a_s)
                    assert v == pytest.approx(v_expected, abs=1e-13)

                    p_expected = p_o[i, d] + (
                        dt * M * v_o[i, d] +
                        dt * M / 2 * Dt * a_l +
                        dt * M**2 / 2 * dt * a_s
                    )
                    assert p == pytest.approx(p_expected, abs=1e-13)
