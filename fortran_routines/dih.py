import numpy as np
import matplotlib.pyplot as plt

##
a = np.array([50, 30, 10], dtype=np.float64)
b = np.array([60, 32, 35], dtype=np.float64)
c = np.array([65, 44, 23], dtype=np.float64)
d = np.array([69, 50, 44], dtype=np.float64)

##
def norm(v):
    s = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    return np.sqrt(s)


def get_dihedral(a, b, c, d):
    b1 = b - a
    b2 = c - b
    b3 = d - c

    b1 /= norm(b1)
    b2 /= norm(b2)
    b3 /= norm(b3)

    # Cross products
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m = np.cross(n1, b2)

    # Dot products
    x = np.dot(n1, n2)
    y = np.dot(m, n2)

    return np.arctan2(y, x)


def new_dihedral(a, b, c, d):
    # Trans dihedral = 0
    # Positive dihedral moving clockwise
    b0 = -(b - a)
    b1 = -(c - b)
    b2 = d - c

    bv = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, bv) * bv
    w = b2 - np.dot(b2, bv) * bv

    x = np.dot(v, w)
    y = np.dot(np.cross(bv, v), w)

    fig = plt.figure(facecolor="white", figsize=(10, 10))
    ax = plt.axes(projection="3d")
    for i in (a, b, c, d):
        ax.scatter(i[0], i[1], i[2], s=100)
    ax.quiver(*b, *b0, arrow_length_ratio=0.05)
    ax.quiver(*c, *b1, arrow_length_ratio=0.05)
    ax.quiver(*c, *b2, arrow_length_ratio=0.05)

    ax.quiver(*b, *v, arrow_length_ratio=0.05, color="r")
    ax.quiver(*b, *w, arrow_length_ratio=0.05, color="g")

    return np.arctan2(y, x)


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
a = np.array([0, 0, 0], dtype=np.float64)
b = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0], dtype=np.float64)
c = np.array([np.sqrt(2) / 2 + 1, np.sqrt(2) / 2, 0], dtype=np.float64)
d = np.array([np.sqrt(2) / 2 + 2, np.sqrt(2), -0.02], dtype=np.float64)
print(np.degrees(get_dihedral(a, b, c, d)))
print(np.degrees(new_dihedral(a, b, c, d)))
# plot_dih(a, b, c, d)
