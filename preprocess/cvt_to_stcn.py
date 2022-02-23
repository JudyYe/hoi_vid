from sys import argv
from tqdm import tqdm
import numpy as np
import cv2
import shutil
import glob
import os
import os.path as osp
import scipy.io as sio


data_dir = '../output/100doh_detectron'
odir = '../output/100doh_detectron/by_obj'
pdir = data_dir + '/by_ppl'

def run(seqname):
    vid_list = [osp.basename(e) for e in glob.iglob(osp.join(odir, 'JPEGImages/%s' % seqname))]
    vid_list = [vid[:-2] for vid in vid_list]
    vid_list = list(set(vid_list))

    for vid in tqdm(vid_list):
        masks = sio.loadmat(
            osp.join(odir, 'Annotations', vid + '_0/00000.mat'), 
            struct_as_record=True)['hand_mask']
        ppl_num = masks.shape[0]
        for p in range(ppl_num):
            shutil.rmtree(osp.join(pdir, 'JPEGImages', vid + '_%d' % p), True)
            shutil.copytree(osp.join(odir, 'JPEGImages', vid + '_0'), 
                            osp.join(pdir, 'JPEGImages', vid + '_%d' % p))

            # mask
            anno_dir = osp.join(pdir, 'Annotations', vid + '_%d' % p)
            os.makedirs(anno_dir, exist_ok=True)
            m_p = ((masks[p] > 0) * 255).astype(np.uint8)
            cv2.imwrite(anno_dir + '/00000.png', m_p)

if __name__ == '__main__':
    seqname = argv[1] if len(argv) >= 2 else '*'
    run(seqname)
