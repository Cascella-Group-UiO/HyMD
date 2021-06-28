How to use `job_arm_profile.sh`:
This script is used to run a short simulation for profiling using Arm Forge (see loaded module in the script).  
`sbatch job_arm_profile config.toml input.h5`
Once over, open a new terminal interfaced with your local computer. Example: (For Betzy)  
`ssh -Y samiransen23@betzy.sigma2.no`  
You may need to load the modules inside `job_arm_profile.sh` once again:
```bash
module load Arm-Forge/20.1.2
module load h5py/2.10.0-foss-2020a-Python-3.8.2
module load pfft-python/0.1.21-foss-2020a-Python-3.8.2
```
Find the `.map` file in the output and:
`map [file-name].map`
It should open a window in `XQuartz` on your local computer using `Arm Forge` in Betzy.

