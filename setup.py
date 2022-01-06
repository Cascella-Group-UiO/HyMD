import setuptools  # noqa: F401, https://stackoverflow.com/a/55358607/4179419
from numpy.distutils.core import setup, Extension


def find_version(path):
    import re
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")


with open('README.md', 'r') as in_file:
    readme = in_file.read()

force_kernels = Extension(
    name="force_kernels",
    sources=[
        "hymd/compute_bond_forces.f90",
        "hymd/compute_bond_forces__double.f90",
        "hymd/compute_angle_forces.f90",
        "hymd/compute_angle_forces__double.f90",
        "hymd/compute_dihedral_forces.f90",
        "hymd/compute_dihedral_forces__double.f90",
        "hymd/dipole_reconstruction.f90",
        "hymd/dipole_reconstruction__double.f90",
    ]
)

setup(
    name="hymd",
    author="Morten Ledum",
    author_email="morten.ledum@gmail.com",
    url="https://github.com/Cascella-Group-UiO/HyMD",
    description="Massively parallel hybrid particle-field MD",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="LGPLv3",
    packages=["hymd"],
    version=find_version("hymd/version.py"),
    ext_modules=[force_kernels],
    setup_requires=[
        "cython",
        "numpy",
        "mpi4py",
    ],
    install_requires=[
        "cython",
        "h5py",
        "mpi4py",
        "mpsort",
        "networkx",
        "numpy",
        "pfft-python",
        "pmesh",
        "sympy",
        "tomli",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",  # noqa: E501
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Fortran",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
