import sys
from ast import parse
import cv2
import numpy as np
import os
import os.path as osp
import glob
import shutil
from tqdm import tqdm

from eval_generic import parse_args
import eval_generic

"""Usage: python eval_generic.py --data_path <path to data_root> --output <some output path>"""


def run(fw_dir, seq):
    bw_dir = fw_dir + '_bw'
    print(fw_dir, bw_dir)

    args.data_path = fw_dir
    args.output = fw_dir + '/Tracks'
    eval_generic.main(args)

    # setup backward track
    setup_inp(fw_dir + '/JPEGImages', bw_dir, fw_dir + '/Tracks', False)
    args.data_path = bw_dir
    args.output = bw_dir + '/Tracks'
    eval_generic.main(args)


def main():
    args = parse_args()
    # # setup forward track
    # print('set up forward track data')
    # setup_inp(fw_dir + '/inp', mask_dir, True)

    # args.data_path = fw_dir + '/inp'
    # args.output = fw_dir + '/out'
    # eval_generic.main(args)

    # setup backward track
    # print('set up backward track data')
    # setup_inp(bw_dir + '/inp', fw_dir + '/out', False)

    args.data_path = bw_dir + '/inp'
    args.output = bw_dir + '/out'
    eval_generic.main(args)

    # vis
    # print('visualize')
    # vis(fw_dir + '/inp', fw_dir + '/out', 'fw', True)
    # vis(bw_dir + '/inp', bw_dir + '/out', 'bw', False)
    

def setup_inp(img_dir, save_dir, mask_dir, forward):
    os.makedirs(osp.join(save_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'Annotations'), exist_ok=True)
    for vid_folder in tqdm(glob.glob(img_dir + '/%s' % args.seq)):
        index = osp.basename(vid_folder)
        dst_img = osp.join(save_dir, 'JPEGImages', index)
        dst_anno =  osp.join(save_dir, 'Annotations', index)
        os.system('rm -r %s' % dst_img)
        os.system('rm -r %s' % dst_anno)

        # make sure the mask is not empty, otherwise, ignore that video ?
        image_list = sorted(glob.glob(osp.join(vid_folder, '*.jpg')))
        if forward:
            mask_src_file = osp.join(mask_dir, index, '00000.png')
        else:
            mask_src_file = osp.join(mask_dir, index, '%05d.png' % (len(image_list) - 1))
        if not osp.exists(mask_src_file) or np.sum((cv2.imread(mask_src_file) > 0)) < 2:
            continue

        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_anno, exist_ok=True)

        if forward:
            shutil.copytree(vid_folder, dst_img, dirs_exist_ok=True)
            # mask
        else:
            image_list.reverse()
            for i in range(len(image_list)):
                shutil.copyfile(image_list[i], osp.join(dst_img, '%05d.jpg' % i))
        
            # mask
        shutil.copyfile(mask_src_file, dst_anno + '/00000.png')


def vis(save_dir, vid_mask_dir, suf, forward=True):
    for vid_folder in glob.iglob(osp.join(save_dir, 'JPEGImages/*')):
        index = osp.basename(vid_folder)

        image_file_list = sorted(glob.glob(osp.join(save_dir, 'JPEGImages', index, '*.jpg')))
        anno_file_list = [osp.join(vid_mask_dir, index, osp.basename(e).replace('.jpg', '.png')) 
            for e in image_file_list]
        if not forward:
            image_file_list.reverse()
            anno_file_list.reverse()

        save_index = '%s/%s_%s' % (vis_dir, index, suf)
        os.makedirs(save_index, exist_ok=True)
        for i, (image_file, anno_file) in enumerate(zip(image_file_list, anno_file_list)):
            image = cv2.imread(image_file)
            mask = (cv2.imread(anno_file) > 0)

            image = np.clip(image + mask * np.array([[0, 0, 125]]), 0, a_max=255)
            cv2.imwrite(osp.join(save_index, '%03d.jpg' % i), image.astype(np.uint8))
        os.system('rm %s.mp4' % save_index)
        cmd = 'ffmpeg -framerate 30 -i {}/%03d.jpg  -c:v libx264 -pix_fmt yuv420p  {}.mp4'.format(save_index, save_index)
        os.system(cmd)


if __name__ == '__main__':
    args = parse_args()
    run(args.folder, args.seq)
    # # make hook
    # fw_dir = '/home/yufeiy2/hoi_vid/output/stcn_output/fw'
    # bw_dir = '/home/yufeiy2/hoi_vid/output/stcn_output/bw'
    # img_dir = '/home/yufeiy2/hoi_vid/output/database/DAVIS/JPEGImages/Full-Resolution'
    # mask_dir = '/home/yufeiy2/hoi_vid/output/database/DAVIS/Annotations/Full-Resolution'
    # vis_dir = '/home/yufeiy2/hoi_vid/output/stcn_output/vis'

    # main()