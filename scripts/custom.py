
import pandas
from jutils import web_utils
import numpy as np
import cv2
import shutil
from collections import defaultdict
from glob import glob
import json
from tqdm import tqdm
import os
import os.path as osp
import argparse

data_dir = '/compute/grogu-2-9/yufeiy2/data/1st'


def view():
    gif_list = sorted(glob(osp.join(data_dir, '*/*.gif')))
    bbox_list = sorted(glob(osp.join(data_dir, '*/*.jpg')))

    cell_list = []
    for gif, bbox in tqdm(zip(gif_list, bbox_list)):
        name = gif.split('/')[-2].split('.')[0]
        line = []
        line.append(name)
        line.append(gif)
        line.append(bbox)
    
        cell_list.append(line)
    
    web_utils.run(osp.join(data_dir, 'vis'), cell_list, width=400)


def ch_format():
    gif_list = sorted(glob(osp.join(data_dir, '*/*.json')))
    seq_list = [osp.dirname(e) for e in gif_list]
    for seq in seq_list:
        print(seq)
        # change all images/x.jpg to images/{04d}.png
        img_list = sorted(glob(osp.join(seq, 'images/*.png')))
        base_name = [int(osp.basename(e).split('.')[0]) for e in img_list]
        base_name = np.argsort(base_name)
        img_list = [img_list[e] for e in base_name]
        print(img_list)

        os.makedirs(osp.join(seq, 'images2'), exist_ok=True)
        print(len(img_list))
        for i, img_path in enumerate(img_list):
            img = cv2.imread(img_path)
            print(img_path)
            cv2.imwrite(osp.join(seq, 'images2/%04d.png' % i), img)
        shutil.rmtree(osp.join(seq, 'images'))
        os.rename(osp.join(seq, 'images2'), osp.join(seq, 'images'))
    return 

def make_web():
    save_dir = '/home/yufeiy2/scratch/result/1st_cache/'
    all_list = sorted(glob(osp.join(save_dir, 'vis/*.mp4')))
    cell_list = [[e] for e in all_list]
    web_utils.run(save_dir+ '/vis_web', cell_list, width=400)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--format', action='store_true')
    parser.add_argument('--make_web', action='store_true')
    return parser.parse_args()


    
if __name__ == '__main__':
    args = get_args()
    if args.view:
        view()
    if args.format:
        ch_format()
    if args.make_web:
        make_web()