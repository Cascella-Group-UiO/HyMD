FROM ubuntu:20.04

# Fix for tzdata installation from
# https://rtfm.co.ua/en/docker-configure-tzdata-and-timezone-during-build/
ENV TZ=Europe/Oslo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    libopenmpi-dev=4.0.3-0ubuntu1 \
    libhdf5-openmpi-dev=1.10.4+repack-11ubuntu1 \
    pkg-config=0.29.1-0ubuntu4 \
    curl=7.68.0-1ubuntu2.2 \
    python3.8=3.8.2-1ubuntu1.2 \
    python3-pip=20.0.2-5ubuntu1

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir cython==0.29.21 && \
    pip3 install --no-cache-dir numpy==1.19.1 && \
    pip3 install --no-cache-dir mpi4py==3.0.3 && \
    pip3 install --no-cache-dir cython==0.29.21 && \
    pip3 install --no-cache-dir networkx==2.5 && \
    pip3 install --no-cache-dir sympy==1.6.2 && \
    pip3 install --no-cache-dir pytest==6.0.1 && \
    pip3 install --no-cache-dir mpsort==0.1.17 && \
    pip3 install --no-cache-dir pfft-python==0.1.21 && \
    pip3 install --no-cache-dir pmesh==0.1.56

COPY . /app
WORKDIR /app

# Compile h5py with MPI support from source. The h5py install script does not
# like to be told where MPI includes are, so first we just put them where
# configure.py expects them to be.
RUN ln -s /usr/lib/x86_64-linux-gnu/openmpi/include/ /usr/include/openmpi

RUN curl -L -o h5py.tar.gz \
    https://github.com/h5py/h5py/archive/6f4c578f78321b857da31eee0ce8d9b1ba291888.tar.gz
RUN tar -xzf h5py.tar.gz && rm h5py.tar.gz && \
  mv h5py-6f4c578f78321b857da31eee0ce8d9b1ba291888 h5py
RUN cd h5py && HDF5_MPI="ON" pip3 install -v . && cd ..
