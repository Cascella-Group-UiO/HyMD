import pytest
import numpy as np
from hymd.input_parser import Config
from hymd.force import (
    compute_bond_forces__plain as compute_bond_forces
)
from hymd.force import (
    compute_angle_forces__plain as compute_angle_forces
)
from hymd.force import (
    compute_dihedral_forces__plain as compute_dihedral_forces
)
from hymd.force import (
    prepare_bonds_old as prepare_bonds
)


def test_prepare_bonds_2(dppc_single):
    indices, bonds, names, molecules, r, CONF = dppc_single
    config = Config(n_steps=1, time_step=0.03, mesh_size=[30, 30, 30],
                    box_size=np.array([13.0, 13.0, 14.0]), sigma=0.5, kappa=1)
    config.bonds = CONF['bond_2']
    bonds_2, _, _, _ = prepare_bonds(molecules, names, bonds, indices, config)
    bonds_2_ind = [b[:2] for b in bonds_2]
    bonds_2_val = [b[2:] for b in bonds_2]

    assert len(bonds_2) == 11
    expected = [[0, 1, 0.47, 1250.0],  # N--P, 0.47nm, 1250.0 kJ/mol nm
                [1, 2, 0.47, 1250.0],  # P--G, 0.47nm, 1250.0 kJ/mol nm
                [2, 3, 0.37, 1250.0],  # G--G, ** 0.37nm **, 1250.0 kJ/mol nm
                [2, 4, 0.47, 1250.0],  # G--C, 0.47nm, 1250.0 kJ/mol nm
                [3, 8, 0.47, 1250.0],  # C--C, 0.47nm, 1250.0 kJ/mol nm
                [4, 5, 0.47, 1250.0],  # C--C, 0.47nm, 1250.0 kJ/mol nm
                [5, 6, 0.47, 1250.0],  # C--C, 0.47nm, 1250.0 kJ/mol nm
                [6, 7, 0.47, 1250.0],  # C--C, 0.47nm, 1250.0 kJ/mol nm
                [8, 9, 0.47, 1250.0],  # C--C, 0.47nm, 1250.0 kJ/mol nm
                [9, 10, 0.47, 1250.0],  # C--C, 0.47nm, 1250.0 kJ/mol nm
                [10, 11, 0.47, 1250.0]]  # C--C, 0.47nm, 1250.0 kJ/mol nm
    for e in expected:
        assert e[:2] in bonds_2_ind
        for ind, val in zip(bonds_2_ind, bonds_2_val):
            if ind == e[:2]:
                assert e[2] == pytest.approx(val[0], abs=1e-13)
                assert e[3] == pytest.approx(val[1], abs=1e-13)


def test_comp_bonds(dppc_single):
    indices, bonds, names, molecules, r, CONF = dppc_single
    config = Config(n_steps=1, time_step=0.03, mesh_size=[30, 30, 30],
                    box_size=np.array([13.0, 13.0, 14.0]), sigma=0.5, kappa=1)
    config.bonds = CONF['bond_2']
    bonds_2, _, _, _ = prepare_bonds(molecules, names, bonds, indices, config)

    expected_energies = np.array([0.24545803261508981,
                                  0.76287125411373635,
                                  2.1147847786016976,
                                  9.3338621118512890,
                                  5.2422610583092810,
                                  1.5660141344942931,
                                  0.32418863877682835,
                                  0.15582792534587750,
                                  6.2278499169351944,
                                  0.65253272104760673,
                                  0.031806604466348753], dtype=np.float64)
    expected_forces_i = np.array([
        [20.998021457611852,    9.0071937483622282,  -9.5707176942411820],
        [34.384875147498597,   12.778365176329645,  -23.697507881539778],
        [17.917253461973676,  -20.426928097726201,  -67.443862458892141],
        [-52.053439897669733,  83.19032187729123,   117.06607117521537],
        [101.22671118485104,   33.178722608565110,   41.928247692168640],
        [-3.8634845559150439, -22.155214019527303,  -58.388828683840543],
        [8.9963139950908992,   12.594088552272277,   23.894075939054996],
        [-2.9353597041563950,   2.4521443094825361, -19.363379484412864],
        [95.85080439063942,    33.391869600494857,   72.575692449944356],
        [23.050482374363451,    8.4211009839435036,  32.079465755314608],
        [5.5323489341501677,    5.4217094311092637,   4.4175438063815795]],
        dtype=np.float64
    )

    expected_forces_j = np.array([
        [-20.998021457611852,  -9.0071937483622282,   9.5707176942411820],
        [-34.384875147498597, -12.778365176329645,   23.697507881539778],
        [-17.917253461973676,  20.426928097726201,   67.443862458892141],
        [52.053439897669733,  -83.19032187729123,  -117.06607117521537],
        [-101.22671118485104, -33.178722608565110,  -41.928247692168640],
        [3.8634845559150439,   22.155214019527303,   58.388828683840543],
        [-8.9963139950908992, -12.594088552272277,  -23.894075939054996],
        [2.9353597041563950,   -2.4521443094825361,  19.363379484412864],
        [-95.85080439063942,  -33.391869600494857,  -72.575692449944356],
        [-23.050482374363451,  -8.4211009839435036, -32.079465755314608],
        [-5.5323489341501677,  -5.4217094311092637,  -4.4175438063815795]],
        dtype=np.float64
    )
    for i, b in enumerate(bonds_2):
        f_bonds = np.zeros(shape=r.shape, dtype=np.float64)
        energy = 0.0
        energy = compute_bond_forces(f_bonds, r, (b,), CONF['L'])
        assert energy == pytest.approx(expected_energies[i], abs=1e-13)
        assert f_bonds[b[0], :] == pytest.approx(expected_forces_i[i], abs=1e-13)  # noqa: E501
        assert f_bonds[b[1], :] == pytest.approx(expected_forces_j[i], abs=1e-13)  # noqa: E501


def test_prepare_bonds_3(dppc_single):
    indices, bonds, names, molecules, r, CONF = dppc_single
    config = Config(n_steps=1, time_step=0.03, mesh_size=[30, 30, 30],
                    box_size=np.array([13.0, 13.0, 14.0]), sigma=0.5, kappa=1)
    config.angle_bonds = CONF['bond_3']
    _, bonds_3, _, _ = prepare_bonds(molecules, names, bonds, indices, config)
    bonds_3_ind = [b[:3] for b in bonds_3]
    bonds_3_val = [b[3:] for b in bonds_3]

    assert len(bonds_3) == 8

    expected = [[1, 2, 3, 120.0, 25.0],  # P--G--G, 120º, 25.0 kJ/mol nm
                [1, 2, 4, 180.0, 25.0],  # P--G--C, 180º, 25.0 kJ/mol nm
                [3, 8, 9, 180.0, 25.0],  # G--C--C, 180º, 25.0 kJ/mol nm
                [2, 4, 5, 180.0, 25.0],  # G--C--C, 180º, 25.0 kJ/mol nm
                [4, 5, 6, 180.0, 25.0],  # C--C--C, 180º, 25.0 kJ/mol nm
                [5, 6, 7, 180.0, 25.0],  # C--C--C, 180º, 25.0 kJ/mol nm
                [8, 9, 10, 180.0, 25.0],  # C--C--C, 180º, 25.0 kJ/mol nm
                [9, 10, 11, 180.0, 25.0]]  # C--C--C, 180º, 25.0 kJ/mol nm
    for e in expected:
        assert e[:3] in bonds_3_ind
        for ind, val in zip(bonds_3_ind, bonds_3_val):
            if ind == e[:3]:
                assert np.radians(e[3]) == pytest.approx(val[0], abs=1e-13)
                assert e[4] == pytest.approx(val[1], abs=1e-13)


def test_comp_angles(dppc_single):
    indices, bonds, names, molecules, r, CONF = dppc_single
    config = Config(n_steps=1, time_step=0.03, mesh_size=[30, 30, 30],
                    box_size=np.array([13.0, 13.0, 14.0]), sigma=0.5, kappa=1)
    config.angle_bonds = CONF['bond_3']
    _, bonds_3, _, _ = prepare_bonds(molecules, names, bonds, indices, config)

    expected_energies = np.array([0.24138227262192161,
                                  12.962077271327919,
                                  2.8815891128087228,
                                  7.4909301390688290,
                                  3.0444037031216604,
                                  1.5356935533508376,
                                  9.2685834423945916,
                                  3.2105275066822720], dtype=np.float64)
    expected_forces_i = np.array([
        [2.4096577139753332,    4.6682763444497457,   6.0136584922358995],
        [-4.9800910270778838, -47.992733040538354,  -33.105104583696715],
        [-15.971987449361599,   5.6577828943870960, -11.122519580320914],
        [15.589371899758191,  -11.591790641852914,  -28.464343403274896],
        [-5.8509283842655613,  27.044650837391227,   -9.8747494843133659],
        [-14.184902829759091,  10.719331754009715,   -0.30921361025473859],
        [-41.940996233256648, -21.778537151795561,    3.5999807991038200],
        [-6.3110894237988617, -22.122793808794096,   10.342190196298951]],
        dtype=np.float64
    )
    expected_forces_j = np.array([
        [-11.393593064392608, -11.244884057123880,  -6.4084691231053164],
        [35.711654217554766,   77.203203600867752,  26.012145349067417],
        [40.447908446946613,  -19.924250597312096,  14.916298415900069],
        [-52.570306735700960,  13.080277480385531,  41.603414667781038],
        [9.8204316768332696,  -53.621653265813158,   5.9073343262146167],
        [25.361507062886922,  -23.245673791098522,  -4.4334005214036809],
        [59.463208645518513,   36.650676356749130, -33.584216759815384],
        [4.7643988951391743,   39.798241357771779, -30.098487542259036]],
        dtype=np.float64
    )
    expected_forces_k = np.array([
        [8.9839353504172745,    6.5766077126741331,   0.39481063086941653],
        [-30.731563190476884, -29.210470560329401,    7.0929592346292987],
        [-24.475920997585010,  14.266467702924999,   -3.7937788355791553],
        [36.980934835942769,   -1.4884868385326178, -13.139071264506143],
        [-3.9695032925677087,  26.577002428421931,    3.9674151580987491],
        [-11.176604233127833,  12.526342037088806,    4.7426141316584198],
        [-17.522212412261865, -14.872139204953569,   29.984235960711562],
        [1.5466905286596875,  -17.675447548977683,   19.756297345960085]],
        dtype=np.float64
    )

    for i, b in enumerate(bonds_3):
        f_angles = np.zeros(shape=r.shape, dtype=np.float64)
        energy = 0.0
        energy = compute_angle_forces(f_angles, r, (b,), CONF['L'])
        assert energy == pytest.approx(expected_energies[i], abs=1e-13)
        assert f_angles[b[0], :] == pytest.approx(expected_forces_i[i], abs=1e-13)  # noqa: E501
        assert f_angles[b[1], :] == pytest.approx(expected_forces_j[i], abs=1e-13)  # noqa: E501
        assert f_angles[b[2], :] == pytest.approx(expected_forces_k[i], abs=1e-13)  # noqa: E501


def test_prepare_bonds_4(alanine_octapeptide):
    indices, bonds, names, molecules, r, CONF = alanine_octapeptide
    config = Config(n_steps=1, time_step=0.03, mesh_size=[30, 30, 30],
                    box_size=np.array([5.0, 5.0, 5.0]), sigma=0.5, kappa=1)
    config.dihedrals = CONF['bond_4']
    _, _, bonds_4, _ = prepare_bonds(molecules, names, bonds, indices, config)
    bonds_4_ind = [b[:4] for b in bonds_4]
    bonds_4_val = [b[4:] for b in bonds_4]
    assert len(bonds_4) == 5

    expected = [
            [0, 2, 4, 6,    np.array([[1 for _ in range(5)], [0 for _ in range(5)]]), 0],  # noqa: E501
            [2, 4, 6, 8,    np.array([[1 for _ in range(5)], [0 for _ in range(5)]]), 0],  # noqa: E501
            [4, 6, 8, 10,   np.array([[1 for _ in range(5)], [0 for _ in range(5)]]), 0],  # noqa: E501
            [6, 8, 10, 12,  np.array([[1 for _ in range(5)], [0 for _ in range(5)]]), 0],  # noqa: E501
            [8, 10, 12, 14, np.array([[1 for _ in range(5)], [0 for _ in range(5)]]), 0],  # noqa: E501
    ]
    for e in expected:
        assert e[:4] in bonds_4_ind
        for ind, val in zip(bonds_4_ind, bonds_4_val):
            if ind == e[:4]:
                # check if this works for arrays?
                assert e[4] == pytest.approx(val[0], abs=1e-13)
                assert e[5] == pytest.approx(val[1], abs=1e-13)


def test_comp_dihedrals(alanine_octapeptide):
    indices, bonds, names, molecules, r, CONF = alanine_octapeptide
    config = Config(
        n_steps=1, time_step=0.03, mesh_size=[30, 30, 30],
        box_size=np.array([5.0, 5.0, 5.0]), sigma=0.5, kappa=1
    )
    config.dihedrals = CONF['bond_4']
    _, _, bonds_4, _ = prepare_bonds(molecules, names, bonds, indices, config)

    expected_energies = np.array([
        5.512306711980792,
        5.501816505737047,
        5.4980072293920745,
        5.50682559136485,
        5.898283428858097],
        dtype=np.float64
    )
    expected_forces_i = np.array([
        [4.167404131528236, 5.964780465237133, 6.027290456833508],
        [-3.88119996455415, -3.364625086216446, -7.922274353931893],
        [1.3438193964197254, 1.721469069878616, 9.164234842488652],
        [-0.5176523283261323, 1.4537046377291065, -9.32030878092003],
        [-1.303516982371019, -1.850843743470206, 5.13282909125343]],
        dtype=np.float64
    )
    expected_forces_j = np.array([
        [-4.374426036878856, -7.901212696584612, -4.597926617980788],
        [5.786331030404696, 4.600082932906689, 7.009782945316923],
        [-1.9629100610700134, -4.100983501069195, -9.046763701062986],
        [2.5517409922373115, -0.20115938578524672, 9.79905918235492],
        [0.9717585303396479, 0.9912206160390491, -4.7894353408271675]],
        dtype=np.float64
    )
    expected_forces_k = np.array([
        [-3.6709627463157175, -1.4254054894919777, -9.345075121810787],
        [-0.5590864711877845, 0.4888617624497804, 10.091901087338766],
        [0.10213174348825493, 3.831271798222488, -9.42529515500928],
        [-4.236386262425887, -4.37955378853861, 8.193186622885671],
        [1.8761151329061132, 3.5095816408063003, -5.2916344246712885]],
        dtype=np.float64
    )
    expected_forces_l = np.array([
        [3.8779846516663383, 3.3618377208394574, 7.915711282958067],
        [-1.346044594662762, -1.7243196091400235, -9.179409678723795],
        [0.5169589211620331, -1.4517573670319093, 9.307824013583614],
        [2.202297598514707, 3.1270085365947504, -8.671937024320561],
        [-1.544356680874742, -2.649958513375143, 4.948240674245026]],
        dtype=np.float64
    )

    for i, b in enumerate(bonds_4):
        f_dihedrals = np.zeros(shape=r.shape, dtype=np.float64)
        energy = 0.0
        energy = compute_dihedral_forces(f_dihedrals, r, (b,), CONF['L'])
        assert energy == pytest.approx(expected_energies[i], abs=1e-13)
        assert f_dihedrals[b[0], :] == pytest.approx(expected_forces_i[i], abs=1e-13)  # noqa: E501
        assert f_dihedrals[b[1], :] == pytest.approx(expected_forces_j[i], abs=1e-13)  # noqa: E501
        assert f_dihedrals[b[2], :] == pytest.approx(expected_forces_k[i], abs=1e-13)  # noqa: E501
        assert f_dihedrals[b[3], :] == pytest.approx(expected_forces_l[i], abs=1e-13)  # noqa: E501
