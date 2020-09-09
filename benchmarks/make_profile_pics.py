import glob
import os
fns=glob.glob('*/*.prof')

for fn in fns:
    fn_out='%s.png'%(fn.split('.')[0])
    os.system('python gprof2dot.py -f pstats %s | dot -Tpng -o %s'%(fn,fn_out))
 
