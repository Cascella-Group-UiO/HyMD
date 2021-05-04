import numpy as np
import matplotlib.pyplot as plt

##
def dihedral_sig(p0, p1, p2, p3):
    # b1 == v2m

    b0 = p0 - p1
    b1 = p1 - p2
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    x = np.dot(b0xb1, b1xb2)
    y = np.dot(b0xb1_x_b1xb2, b1) / np.linalg.norm(b1)

    # fig = plt.figure(facecolor="white", figsize=(10, 10))
    # ax = plt.axes(projection="3d")

    # for i in (a, b, c, d):
    #    ax.scatter(i[0], i[1], i[2], s=100)
    # ax.quiver(*b, *b0, arrow_length_ratio=0.05)
    # ax.quiver(*c, *b1, arrow_length_ratio=0.05)
    # ax.quiver(*c, *b2, arrow_length_ratio=0.05)

    # ax.quiver(*b, *b0xb1, arrow_length_ratio=0.05, color="r")
    # ax.quiver(*b, *b1xb2, arrow_length_ratio=0.05, color="g")

    return np.degrees(np.arctan2(y, x))


def dihedral_norm(p0, p1, p2, p3):
    # b1 == v2m

    b0 = p0 - p1
    b1 = p1 - p2
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1 /= np.linalg.norm(b0xb1)
    b1xb2 /= np.linalg.norm(b1xb2)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    x = np.dot(b0xb1, b1xb2)
    y = np.dot(b0xb1_x_b1xb2, b1) / np.linalg.norm(b1)
    fig = plt.figure(facecolor="white", figsize=(10, 10))
    ax = plt.axes(projection="3d")

    for i in (a, b, c, d):
        ax.scatter(i[0], i[1], i[2], s=100)
    ax.quiver(*b, *b0, arrow_length_ratio=0.05)
    ax.quiver(*c, *b1, arrow_length_ratio=0.05)
    ax.quiver(*c, *b2, arrow_length_ratio=0.05)

    ax.quiver(*b, *b0xb1, arrow_length_ratio=0.05, color="r")
    ax.quiver(*b, *b1xb2, arrow_length_ratio=0.05, color="g")

    return np.degrees(np.arctan2(y, x))


def new_dihedral(p1, p2, p3, p4):
    # Angle calculation, taken from:
    # https://stackoverflow.com/a/34245697
    # Trans dihedral = 180
    # Positive dihedral moving clockwise
    b0 = p1 - p2
    b1 = p2 - p3
    b2 = p4 - p3
    bn = np.linalg.norm(b1)

    bv = b1 / bn
    v = b0 - np.dot(b0, bv) * bv
    w = b2 - np.dot(b2, bv) * bv

    x = np.dot(v, w)  # cos(phi)
    y = np.dot(np.cross(bv, v), w)  # sin(phi)
    return np.degrees(np.arctan2(y, x))


def dV_dphi(p1, p2, p3, p4, ci, di, phi):
    # Force calculation, taken from:
    # A. Blondel, M. Karplus, JCC Vol 17, No 9, 1132-1141 (1996)
    b0, b1, b2, bn = prepare_vectors(p1, p2, p3, p4)

    # Cross products
    a = np.cross(b0, b1)
    b = np.cross(b2, b1)
    # Squared norms
    a2 = np.dot(a, a)
    b2 = np.dot(b, b)
    # Dot products
    fg = np.dot(b0, b1)
    hg = np.dot(b2, b1)
    s = a * fg / (a2 * bn) - b * hg / (b2 * bn)
    # Forces on particles
    df = 0
    for i in range(len(ci)):
        df += i * ci[i] * np.sin(i * phi + di[i])

    F1 = df * -bn * a / a2
    F4 = df * bn * b / b2
    F2 = df * s - F1
    F3 = df * -s - F4
    return F1, F2, F3, F4


def prepare_vectors(p1, p2, p3, p4):
    b0 = p1 - p2
    b1 = p2 - p3
    b2 = p4 - p3

    for i, b in enumerate(box):
        b0[i] -= b * np.rint(b * b0[i])
        b1[i] -= b * np.rint(b * b1[i])
        b2[i] -= b * np.rint(b * b2[i])

    bn = np.linalg.norm(b1)
    return b0, b1, b2, bn


#     fig = plt.figure(facecolor="white", figsize=(10, 10))
#     ax = plt.axes(projection="3d")
#     for i in (a, b, c, d):
#         ax.scatter(i[0], i[1], i[2], s=100)
#     ax.quiver(*b, *b0, arrow_length_ratio=0.05)
#     ax.quiver(*c, *b1, arrow_length_ratio=0.05)
#     ax.quiver(*c, *b2, arrow_length_ratio=0.05)
#
#     ax.quiver(*b, *v, arrow_length_ratio=0.05, color="r")
#     ax.quiver(*b, *w, arrow_length_ratio=0.05, color="g")


def plot_dih(a, b, c, d):
    b1 = b - a
    b2 = c - b
    b3 = d - c

    fig = plt.figure(facecolor="white", figsize=(10, 10))
    ax = plt.axes(projection="3d")
    for i in (a, b, c, d):
        ax.scatter(i[0], i[1], i[2], s=100)
    ax.quiver(*b, *b1, arrow_length_ratio=0.05)
    ax.quiver(*b, *b2, arrow_length_ratio=0.05)
    ax.quiver(*c, *b3, arrow_length_ratio=0.05)


##
a = np.array([-1, 0, 0], dtype=np.float64)
b = np.array([0, 1, 0], dtype=np.float64)
c = np.array([2, 1, 0], dtype=np.float64)
d = np.array([2.5, 1, 0.5], dtype=np.float64)
print(dihedral_sig(a, b, c, d))
print(dihedral_norm(a, b, c, d))
print(new_dihedral(a, b, c, d))
##
a = np.array([10, 10, 10], dtype=np.float64)
print(np.dot(a, a))
print(np.linalg.norm(a))
print(np.linalg.norm(a, ord=1))
