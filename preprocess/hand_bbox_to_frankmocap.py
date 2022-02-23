"""
Output: 
    bboxes saved in box_dir/ ready for frankmocap to consume
Input:
    per-frame bboxes from find_hand.py ${seq}
"""
import glob
import json
import os
import os.path as osp
import pickle
from sys import argv
from flask import jsonify
import numpy as np

odir = '../output/100doh_detectron/by_obj/'
fdir = '../output/100doh_detectron/by_seq/'

def load_hand_bboxes(seqname):
    hand_meta = pickle.load(
        open(osp.join(odir, 'VidAnnotations', '%s_%d/hand_inds_side.pkl' % (seqname, 0)), 'rb'))
    hand_box = hand_meta['hand_box']

    hand_side = hand_meta['handside']
    print(hand_side)
    assert len(hand_box) == len(hand_side)
    return hand_box, hand_side


def to_frankmocap(seqname):
    hand_bbox, hand_side = load_hand_bboxes(seqname)  # []

    for f in range(hand_bbox.shape[1]):
        json_obj = {'image_path': osp.join(odir, 'JPEGImages', '%s_%d/%05d.jpg' % (seqname, 0, f)), 
            'hand_bbox_list': [], 'body_bbox_list': []}

        for p in range(hand_bbox.shape[0]):
            hand = 'right_hand' if hand_side[p] == '1' else 'left_hand'
            x1, y1, x2, y2 = hand_bbox[p, f]
            xywh = [x1, y1, x2 - x1, y2 - y1]
            # hack hands that lost track
            if xywh[2] == 0:
                xywh[2] = 10
            if xywh[3] == 0:
                xywh[3] = 10

            json_obj['hand_bbox_list'].append(
                {hand: xywh}
            )
            json_obj['body_bbox_list'].append(xywh)

        json_file = osp.join(fdir, 'hand_box', seqname, '%05d.json' % f)
        os.makedirs(osp.dirname(json_file), exist_ok=True)
        with open(json_file, 'w') as fp:
            json.dump(json_obj, fp, indent=4)


if __name__ == '__main__':
    seqname = argv[1] if len(argv) >= 2 else '*'
    vid_list = glob.glob(osp.join(odir, 'JPEGImages', seqname))
    vid_list = [osp.basename(e) for e in vid_list]
    
    for seqname in vid_list:
        to_frankmocap(seqname)
    