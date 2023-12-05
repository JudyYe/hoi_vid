import numpy as np
import os.path as osp
import subprocess

import glob
import os
from sys import argv

skip = bool(argv[1])


# raw_dir = os.environ['RAWDIR']
# odir = os.environ['ODIR']
raw_dir='/private/home/yufeiy2/scratch/result/wild/raw'    # input sequence path
odir = '/private/home/yufeiy2/scratch/result/wild/raw_cache/by_obj'

# vid_list = [os.path.basename(e) for e in glob.glob('/home/yufeiy2/hoi_vid/output/100doh_clips/*')]
vid_list = [e.split('/')[-2] for e in glob.glob(osp.join(f'{raw_dir}', '*/*.json'))]
np.random.shuffle(vid_list)
print(f'{raw_dir}/*/*.json')
for vid in vid_list:
    print(vid)
    if skip:
        if len(glob.glob(f'{odir}/Meta/%s*.json' % vid)) > 0:
            continue
    lock_file = f'{odir}/Locks/{vid}'
    try:
        os.makedirs(lock_file)
    except FileExistsError:
        if skip:
            continue
    print(vid)
    # subprocess.call(["./vos.sh", vid], shell=True)
    subprocess.call(["./preprocess/vos.sh", vid])
    # break

    # cmd = 'bash eval_vis.py %s' % vid
    # print(cmd)
    # os.system(cmd)

    os.system('rm -rf %s' % lock_file)