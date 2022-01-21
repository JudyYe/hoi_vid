from sys import argv
import imageio
import json
from tqdm import tqdm
import numpy as np
import cv2
import shutil
import glob
import os
import os.path as osp
import scipy.io as sio
import ffmpeg_utils

data_dir = '../output/100doh_detectron'
odir = '../output/100doh_detectron/by_obj'
pdir = data_dir + '/by_ppl'
va_dir = odir + '/VidAnnotations'
meta_dir = odir + '/Meta'

vis_dir = data_dir + '/vis/'



def evaluate_fw_bw(track_dir):
    vid_list = sorted(glob.glob(osp.join(track_dir, seqname)))
    vid_list = [osp.basename(vid) for vid in vid_list]
    iou_dict = {}
    print('evaluating ', osp.join(track_dir, seqname))
    for vid in tqdm(vid_list):
        fw_file = track_dir + vid + '/00000.png'
        num_frame = len(glob.glob(osp.join(va_dir, vid, '*.png')))
        bw_file = track_dir.replace('/Tracks',  '_bw/Tracks/') + vid + '/%05d.png' % (num_frame - 1)
        if not osp.exists(bw_file):
            print(bw_file)
            iou = 0
        elif not osp.exists(fw_file):
            print('????', fw_file)
            iou = 0
        else:
            fw = cv2.imread(fw_file) > 0
            bw = cv2.imread(bw_file) > 0
            iou = (fw & bw).sum() / ((fw | bw).sum() + 1)
        iou_dict[vid] = iou
    return iou_dict


def visualize(seqname):
    vid_list = [osp.basename(e) for e in sorted(glob.glob(odir + '/VidAnnotations/%s' % seqname))]
    print('visualize')
    for vid in tqdm(vid_list):
        frame_list = sorted(glob.glob(odir + '/VidAnnotations/%s/*.png' % vid))
        try:
            ppl_mask = np.load(odir + '/VidAnnotations/%s/ppl.npy' % vid)
        except FileNotFoundError:
            ppl_mask = []

        image_list = []
        for f, mask_file in enumerate(frame_list):
            image = imageio.imread(osp.join(odir, 'JPEGImages', vid, '%05d.jpg' % f))
            obj_mask = (imageio.imread(mask_file) > 0) 
            image = draw_mask(image, obj_mask, (255, 255, 255))
            for p in range(len(ppl_mask)):
                image = draw_mask(image, ppl_mask[p, f], (0, 255, 0))
            image_list.append(image)
        meta = json.load(open(osp.join(meta_dir, vid + '.json')))

        ppl_score = [v for k, v in meta.items() if k != 'obj']
        ppl_score = np.sum(ppl_score) / (len(ppl_score) + 1)
        score = meta['obj']
        save_file = osp.join(vis_dir,  '%02d_%02d_%s' % (int(score * 100), int(ppl_score * 100), vid))
        ffmpeg_utils.write_mp4(image_list, save_file)


def draw_mask(image, mask, color, r=0.7):
    if mask.ndim == 2:
        mask = mask[..., None]
    image = image * 0.9 * (1 - mask) + mask * (image * r + (1 - r) *np.array([color]))
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image
            


def evaluate(seqname):
    obj_iou_list = evaluate_fw_bw(odir + '/Tracks/')
    ppl_iou_list = evaluate_fw_bw(pdir + '/Tracks/')

    vid_to_ppl = {}
    for p in ppl_iou_list:
        vid_index = p[:-2]
        if not vid_index in vid_to_ppl:
            vid_to_ppl[vid_index] = []
        vid_to_ppl[vid_index].append(p)
    
    for obj_ind in obj_iou_list:
        vid_index = obj_ind[:-2]
        iou_score = {'obj': obj_iou_list[obj_ind]}
        for p in vid_to_ppl.get(vid_index, []):
            iou_score[p[-2:]] = ppl_iou_list[p]
        os.makedirs(meta_dir, exist_ok=True)
        json.dump(iou_score, open(meta_dir +'/%s.json' % obj_ind, 'w'), indent=4)


def merge(seq):
    vid_list = sorted(glob.glob(odir + '/Tracks/%s' % seq))
    vid_list = [osp.basename(vid) for vid in vid_list]
    for vid in tqdm(vid_list):
        # fw track of object
        if osp.exists(osp.join(va_dir, vid)): shutil.rmtree(osp.join(va_dir, vid))
        shutil.copytree(odir + '/Tracks/' + vid, osp.join(va_dir, vid))
        # replace first frame 
        num_frame = len(glob.glob(osp.join(va_dir, vid, '*.png')))
        bw_file = odir + '_bw/Tracks/' + vid + '/%05d.png' % num_frame
        if osp.exists(bw_file):
            shutil.copyfile(bw_file, osp.join(va_dir, vid, '%05d.png' % 0))

        # ppl
        ppl = vid[:-2]
        ppl_list = sorted(glob.glob(osp.join(pdir, 'Tracks', ppl + '_*')))
        ppl_mask = []
        for ppl_dir in ppl_list:
            mask_file_list = sorted(glob.glob(ppl_dir + '/*.png'))
            mask_list = [cv2.imread(e)[..., 0] > 0 for e in mask_file_list]
            bw_file = pdir + '_bw/Tracks/' + vid + '/%05d.png' % num_frame
            if osp.exists(bw_file):
                mask_list[0] = cv2.imread(bw_file)[..., 0] > 0

            ppl_mask.append(mask_list)  # P, T, H, W
        if len(ppl_mask) > 0:
            ppl_mask = np.array(ppl_mask)
            print('ppl mask', ppl_mask.shape)
            np.save(osp.join(va_dir, vid, 'ppl.npy'), ppl_mask)


if __name__ == '__main__':
    seqname = argv[1] if len(argv) >= 2 else '*'

    merge(seqname)
    evaluate(seqname)
    visualize(seqname)