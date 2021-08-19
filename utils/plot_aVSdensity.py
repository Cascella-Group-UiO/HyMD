import matplotlib.pyplot as plt
import fileinput
import numpy as np
import math
from argparse import ArgumentParser
import os
import h5py

HYMD_PATH="/Users/samiransen23/hymdtest"
INPUT_PATH="/Users/samiransen23/hymdtest"

def PLOT_aVSdensity(sigma, a, folder_path):
    V = np.zeros((len(sigma),len(a)))
    density = np.zeros((len(sigma),len(a)))
    for ii in range(len(sigma)):
        for i in range(len(a)):
            #file_path = os.path.abspath("/Users/samiransen23/hymdtest/test_aparametrization/a="+a[i]+"/sim.h5")
            #h5md_file = open_h5md_file(file_path)
            file_path = os.path.abspath(folder_path+"/sigma="+sigma[ii]+"/a="+a[i]+"/sim.h5")
            f = h5py.File(file_path, "r")
            N = len(list(f["particles/all/species"]))
            V[ii,i] = np.prod(f["particles/all/box/edges/value"][-1])
            density[ii,i] = N/V[ii,i] * 0.11955 #gm/cc

    ##PLOTS
    [float(i) for i in a]
    print('a:',a)
    for ii in range(len(sigma)):
        plt.xlabel("a")
        plt.ylabel("density (gm/cm^3)")
        plt.axhline(y=1, color='y', linestyle='-', label='Natural density line')
        plt.text(11.5,1,'density = 1 gm/cm^3', fontsize=12, va='center', ha='center', backgroundcolor='w')
        plt.plot(a, density[ii], marker='o', label="a vs density")
        plt.title('sigma='+sigma[ii])
        plt.legend()
        plt.show()


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--confscript",default=None, help="water/CONF.py")
    ap.add_argument("--path",default=None, help="aparametrisation/")
    ap.add_argument("--plot_aVSdensity", default=False, help="True")
    ap.add_argument("--sigma", type=str, nargs="+", default=None, help="0.238 0.338 0.438")
    ap.add_argument("--a", type=str, nargs="+", default=None, help="9.25 9.30")
    args = ap.parse_args()
    if args.plot_aVSdensity is not None: PLOT_aVSdensity(args.sigma, args.a, args.path)
