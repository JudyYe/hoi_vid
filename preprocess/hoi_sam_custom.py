# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import shutil
import scipy.io as sio
import json
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import os
import os.path as osp
import numpy as np
import tqdm
import imageio
from  matplotlib import pyplot as plt
raw_dir = os.environ['RAWDIR']
odir = os.environ['ODIR']

from sys import argv
from segment_anything import SamPredictor, sam_model_registry

ext = '.jpg'
def setup_model():
    sam = sam_model_registry["default"](checkpoint="/private/home/yufeiy2/scratch/pretrain/sam/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    return predictor

def test(seqname, predictor: SamPredictor):
    # detect the first frame, move object seq to odir 
    # rename seq to %05d from 0
    imgdir= '%s/JPEGImages/%s'%(odir,seqname)
    maskdir='%s/Annotations/%s'%(odir,seqname)

    path = osp.join(raw_dir, seqname, f'images/{0:04d}{ext}')
    try:
        obj_box, hand_box = find_gt_boxes(osp.join(raw_dir, seqname))
    except FileNotFoundError:
        print('skip', raw_dir, seqname)
        return 
    img = imageio.imread(path) # first frame

    predictor.set_image(img, 'RGB')

    obj_masks, obj_score, _ = predictor.predict(box=obj_box, )
    hand_masks, hand_score, _ = predictor.predict(box=hand_box, )

    print(img.shape, obj_masks.shape, hand_masks.shape, obj_score, hand_score)

    ind = np.argmax(obj_score)
    obj_masks = obj_masks[ind:ind+1]
    obj_score = obj_score[ind:ind+1]
    ind = np.argmax(hand_score)
    hand_masks = hand_masks[ind:ind+1]
    hand_score = hand_score[ind:ind+1]

    out_pref = osp.join(odir, 'vis_debug/vis_%s' % seqname,)
    vis(img, hand_masks, obj_masks, hand_box, obj_box, out_pref + '_sam.png')

    hand_masks = ((hand_masks > 0) * 255).astype(np.uint8)
    obj_masks = ((obj_masks > 0) * 255).astype(np.uint8)

    for o in range(len(obj_masks)):
        os.makedirs(maskdir + '_%d' % o, exist_ok=True)
        cv2.imwrite(maskdir + '_%d/%05d.png' % (o, 0), obj_masks[o])
        print(hand_masks.shape, obj_masks[o].shape, obj_masks.dtype,)
        sio.savemat(maskdir + '_%d/%05d.mat' % (o, 0), 
            {'hand_mask': hand_masks, 'hand_box': hand_box[None],
            'obj_box': obj_box})

        dst_folder = imgdir + '_%d' % o
        if osp.exists(dst_folder): shutil.rmtree(dst_folder)
        os.makedirs(dst_folder)
        # src_folder = osp.join(raw_dir, '%s/frames/*.jp*' % seqname)
        src_folder = osp.join(raw_dir, '%s/images/*.*g' % seqname)

        for i,src_path in enumerate(sorted(glob.glob(src_folder))):
            # read jpg and save as png
            img = cv2.imread(src_path)
            cv2.imwrite(osp.join(dst_folder, '%05d.jpg' % i), img)



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def vis(image, hand_mask, obj_mask, hand_box, obj_box, save_path='test.png'):
    hand_box = hand_box.astype(np.int32)
    obj_box = obj_box.astype(np.int32)
    cv2.rectangle(image, (hand_box[0], hand_box[1]), (hand_box[2], hand_box[3]), (0, 255, 0), 2)
    cv2.rectangle(image, (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3]), (0, 0, 255), 2)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(hand_mask[0], plt.gca(), True)
    show_mask(obj_mask[0], plt.gca(), True)
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

def find_gt_boxes(seq_dir):  
    """
    filter_field: targetobject / hand
    """
    boxes = json.load(open(osp.join(seq_dir, 'bbox.json')))
    obj_box = np.array(boxes['obj'])
    hand_box = np.array(boxes['hand'])
    # hand_box = pad_box(hand_box, 0.2)
    # obj_box = pad_box(obj_box, 0.1)
    return obj_box, hand_box

def pad_box(box, pad=0.1):
    """
    box: x1y1x2y2
    """
    box = box.copy()
    w = box[2] - box[0]
    h = box[3] - box[1]
    m = min(w, h)
    box[0] -= m * pad
    box[1] -= m * pad
    box[2] += m * pad
    box[3] += m * pad
    return box

if __name__ == '__main__':
    predictor = setup_model()
    seqname = argv[1] if len(argv) >= 2 else '*'
    vid_list = glob.glob(osp.join(raw_dir, seqname))
    vid_list = [osp.basename(e) for e in vid_list]
    print('Running detectron2 to extract first frame')
    print('save to ', odir)
    for vid in tqdm.tqdm(vid_list):
        test(vid, predictor)
    print('!save to ', odir)