import matplotlib.pyplot as plt
import fileinput
import numpy as np
import math
from argparse import ArgumentParser
import os
import h5py

def PLOT_aVSdensity(sigma, a, folder_path):
    '''
    folder_path:   path to folder containing sigma and a valued folders.
                   Example: /Users/samiransen23/aparametrization
                   >ls /Users/samiransen23/aparametrization
                   sigma=0.238 sigma=0.338 sigma=0.438
                   >ls /Users/samiransen23/aparametrization/sigma\=0.238
                   a=9.25 a=9.30 a=9.35
                   a=9.25 should contain the trajectory sim.h5
    '''
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
            f.close()

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

def PLOT_mVSvolume(m, folder_path):
    '''
    This function plots a list of chosen density factors m versus
    the equilibrium volume under constant ext pressure.
    folder_path:   path to folder containing m valued folders.
                   Example: /Users/samiransen23/out
                   >ls /Users/samiransen23/out
                   m=1.0 m=1.1 m=1.2 m=1.3
                   m=1.0 should contain the trajectory sim.h5
    '''
    volume = np.zeros(len(m))
    density = np.zeros(len(m))
    areapl = np.zeros(len(m))
    for ii in range(len(m)):
        file_path = os.path.abspath(folder_path+"/m="+m[ii]+"/sim.h5")
        f = h5py.File(file_path, "r")
        volume[ii] = np.prod(f["particles/all/box/edges/value"][-1])
        N = len(list(f["particles/all/species"]))
        density[ii] = N/volume[ii] * 0.11955 #gm/cc
        areapl[ii] = np.prod(f["particles/all/box/edges/value"][-1][0:2])/264
        f.close()

    ##PLOTS
    [float(i) for i in m]
    plt.xlabel("m")
    plt.ylabel("volume (nm^3)")
    plt.plot(m, volume, marker='o', label="m vs volume")
    plt.legend()
    plt.show()
    plt.xlabel("m")
    plt.ylabel("density (gm/cm^3)")
    plt.plot(m, density, marker='o', label="m vs density")
    plt.legend()
    plt.show()
    plt.xlabel("m")
    plt.ylabel("area per lipid (nm^2)")
    plt.plot(m, areapl, marker='o', label="m vs area per lipid")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--confscript",default=None, help="water/CONF.py")
    ap.add_argument("--path",default=None, help="path of folder containing folders as reqd. See inside function")
    ap.add_argument("--plot_aVSdensity", default=False, help="True")
    ap.add_argument("--plot_mVSvolume", default=False, help="True")
    ap.add_argument("--sigma", type=str, nargs="+", default=None, help="0.238 0.338 0.438")
    ap.add_argument("--a", type=str, nargs="+", default=None, help="9.25 9.30 9.35")
    ap.add_argument("--m", type=str, nargs="+", default=None, help="1.0 1.1 1.2")
    args = ap.parse_args()
    if args.plot_aVSdensity: PLOT_aVSdensity(args.sigma, args.a, args.path)
    if args.plot_mVSvolume: PLOT_mVSvolume(args.m, args.path)
