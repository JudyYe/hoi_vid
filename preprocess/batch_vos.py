import subprocess

import glob
import os
from sys import argv

skip = bool(argv[1])

raw_dir = os.environ['RAWDIR']
odir = os.environ['ODIR']
# vid_list = [os.path.basename(e) for e in glob.glob('/home/yufeiy2/hoi_vid/output/100doh_clips/*')]
vid_list = [e.split('/')[-2] for e in glob.glob(f'{raw_dir}/*/*.json')]


for vid in vid_list:
    print(vid)
    if skip == 1:
        if len(glob.glob(f'{odir}/Meta/%s*.json' % vid)) > 0:
            continue
    print(vid)
    # subprocess.call(["./vos.sh", vid], shell=True)
    subprocess.call(["./preprocess/vos.sh", vid])
    # break

    # cmd = 'bash eval_vis.py %s' % vid
    # print(cmd)
    # os.system(cmd)

