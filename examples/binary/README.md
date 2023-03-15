To make a binary phase system, we just make a box of side length 25 nm and fill it with 130686 water beads (this number corresponds to natural density, so it is optional). We use a CG input of `water.gro` from CGMartini website.

```bash
gmx solvate -cs water.gro -o allA.gro -box 25 25 25 -maxsol 130686
```
We run the following script to rename all waterbeads above half the z-axis as B and the rest as A.

```python
import MDAnalysis as mda
  u = mda.Universe('allA.gro')
for atoms in u.atoms:
    if(atoms.position[2] > 125):
        atoms.residue.resname = 'B'
    else:
        atoms.name = 'A'
        atoms.residue.resname = 'A'
      
    print(atoms.resid)

u.atoms.write('AB.gro')
```

We now have a biphasic system with the AB boundary in the XY plane at z = 12.5 nm
