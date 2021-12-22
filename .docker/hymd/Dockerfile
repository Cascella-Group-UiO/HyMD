FROM ubuntu:20.04

# Fix for tzdata installation from
# https://rtfm.co.ua/en/docker-configure-tzdata-and-timezone-during-build/
ENV TZ=Europe/Oslo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y     \
    libopenmpi-dev                           \
    libhdf5-openmpi-dev                      \
    pkg-config                               \
    curl                                     \
    python3.8                                \
    python3-pip                              \
    python3-mpi4py                           \
    python3-numpy

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir cython numpy mpi4py cython && \
    python3 -m pip install networkx sympy pytest pytest-mpi mpsort pfft-python pmesh tomli

ENV CC=mpicc
ENV HDF5_MPI="ON"
RUN python3 -m pip install --no-cache-dir --no-binary=h5py h5py

COPY . /app
WORKDIR /app
