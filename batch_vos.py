import subprocess

import glob
import os
from sys import argv

skip = bool(argv[1])

vid_list = [os.path.basename(e) for e in glob.glob('/home/yufeiy2/hoi_vid/output/100doh_clips/*')]
for vid in vid_list:
    print(vid)
    if skip == 1:
        if len(glob.glob('../output/100doh_detectron/by_obj/Meta/%s*.json' % vid)) > 0:
            continue
    print(vid)
    # subprocess.call(["./vos.sh", vid], shell=True)
    subprocess.check_call(["./vos.sh", vid])
    # break

    # cmd = 'bash eval_vis.py %s' % vid
    # print(cmd)
    # os.system(cmd)

