#!/bin/bash
set -x
set -e
################################################################################
# File:    buildDocs.sh
# Purpose: Script that builds our documentation using sphinx and updates GitHub
#          Pages. This script is executed by:
#            .github/workflows/docs_pages_workflow.yml
#
# Authors: Michael Altfield <michael@michaelaltfield.net>
# Created: 2020-07-17
# Updated: 2020-07-17
# Version: 0.1
################################################################################

###################
# INSTALL DEPENDS #
###################

sudo apt-get update
sudo apt-get -y install git make rsync wget pkg-config libhdf5-serial-dev python3-numpy python3-h5py python3-mpi4py python3-pip

python3 -m pip install -U cython numpy mpi4py
python3 -m pip install -r requirements.txt
python3 -m pip install -r docs/docs_requirements.txt
python3 -m pip install .

#####################
# DECLARE VARIABLES #
#####################

pwd
ls -lah
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

##############
# BUILD DOCS #
##############

# build our documentation with sphinx (see docs/conf.py)
# * https://www.sphinx-doc.org/en/master/usage/quickstart.html#running-the-build
make -C docs clean
make -C docs html

# exit cleanly
exit 0
