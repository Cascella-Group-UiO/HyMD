FROM ubuntu:latest

# Fix for tzdata installation from
# https://rtfm.co.ua/en/docker-configure-tzdata-and-timezone-during-build/
ENV TZ=Europe/Oslo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update                  \
    && apt-get install -y           \
    libopenmpi-dev                  \
    libhdf5-openmpi-dev             \
    pkg-config                      \
    curl                            \
    python3-pip                     \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip                                       \
    && python3 -m pip install --no-cache-dir --upgrade cython numpy mpi4py     \
    && python3 -m pip install mpsort networkx pfft-python pmesh sympy tomli

RUN CC="mpicc" HDF5_MPI="ON" python3 -m pip install --no-cache-dir --no-binary=h5py h5py

COPY . /app
WORKDIR /app
