"""
Output:
    distribute hand detection (mocap results) to each obj seq
    and copy to DAVIS (for lasr)
Input:
    per-frame mocap results
"""

import json
import os
import os.path as osp
import pickle
from sys import argv
import numpy as np
from glob import glob
import shutil

odir = '../output/100doh_detectron/by_obj/'
fdir = '../output/100doh_detectron/by_seq/'

lasr_dir = '../output/database/DAVIS/Mocap/Full-Resolution/'
def dist_to_obj(seqname):
    num_obj = len(glob(osp.join(odir, 'VidAnnotations', seqname + '*')))
    for o in range(num_obj):
        dst_dir = osp.join(odir, 'mocap_seq', '%s_%d' % (seqname, o))
        shutil.rmtree(dst_dir, True)
        src_dir = osp.join(fdir, 'mocap', seqname, )
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        dst_dir = osp.join(lasr_dir, 'r%s_%d' % (seqname, o))
        shutil.rmtree(dst_dir, True)
        src_dir = osp.join(fdir, 'mocap', seqname, )
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


if __name__ == '__main__':
    seqname = argv[1] if len(argv) >= 2 else '*'
    vid_list = glob.glob(osp.join(odir, 'JPEGImages', seqname))
    vid_list = [osp.basename(e) for e in vid_list]
    
    for seqname in vid_list:
        dist_to_obj(seqname)
