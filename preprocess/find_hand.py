"""
Output: 
    all hand boxes (P, T, 4) for each ojbect sequence
    correspondence bbox (l, ) for each ojbect 
Input:
    hand_mask per frame (P, T, H, W)
    annotation.xml
"""

import pickle
import cv2
import imageio
import glob
import os
import os.path as osp
import xml.etree.ElementTree as ET
from sys import argv
import scipy.io as sio
import numpy as np
from detectron2.structures import pairwise_iou, Boxes
import torch
from jutils import image_utils

data_dir = '../data/100doh/'
odir = '../output/100doh_detectron/by_obj/'

            # {'hand_mask': hand_mask, 'hand_box': gt_boxes[hand_inds],
            # 'obj_box': gt_boxes[obj_inds[o]]})

def load_anno(index):
    """Images"""
    path = os.path.join(data_dir, 'Annotations', index + '.xml')
    if not osp.exists(path):
        print (path)
        return None
    with open(path, 'r') as fid:
        root = ET.parse(fid).getroot()
    return root


def extract_bbox(box, pref=''):
    def cvt(x):
        if x is None or x == 'None':
            return -1
        return float(x)
    x1 = cvt(box.find('%sxmin' % pref).text)
    x2 = cvt(box.find('%sxmax' % pref).text)
    y1 = cvt(box.find('%symin' % pref).text)
    y2 = cvt(box.find('%symax' % pref).text)
    return [x1, y1, x2 ,y2]


def find_boxes(root, filter_field=None):  
    """
    filter_field: targetobject / hand
    """
    gt_boxes, is_object, hand_side, contact_bbox = [], [], [], []
    for obj in root.findall('object'):
        cls = obj.find('name')
        if filter_field is None or cls.text == filter_field:
            gt_bbox = extract_bbox(obj.find('bndbox'))
            gt_boxes.append(gt_bbox)
            is_object.append(cls.text == 'targetobject')
            hand_side.append(obj.find('handside').text)
            contact_bbox.append(extract_bbox(obj, 'obj'))
    gt_boxes = np.array(gt_boxes)
    is_object = np.array(is_object)
    hand_side = np.array(hand_side)
    contact_bbox = np.array(contact_bbox)
    return gt_boxes, is_object, hand_side, contact_bbox


def find_corsp_hand(seqname):
    root = load_anno(seqname)
    gt_boxes, is_object, hand_side, contact_box = find_boxes(root)


    hand_inds = np.where(is_object == 0)[0]
    obj_inds = np.where(is_object == 1)[0]
    contact_box = contact_box[hand_inds]
    hand_side = hand_side[hand_inds]
    hand_boxes = gt_boxes[hand_inds]
    obj_bbox = gt_boxes[obj_inds]

    adj = build_graph(hand_boxes, obj_bbox, contact_box)

    hand_mask = np.load(osp.join(odir, 'VidAnnotations', '%s_%d/ppl.npy' % (seqname, 0)))
    all_hand_bbox = []
    bad_tracks = {}
    for p in range(len(hand_mask)):
        all_hand_bbox.append([hand_boxes[p]])
        for f in range(1, len(hand_mask[p])):
            box = image_utils.mask_to_bbox(hand_mask[p, f], 'med', rate=1.5)
            if np.all(box == 0):
                bad_tracks[p] = 1
            all_hand_bbox[-1].append(box)
    all_hand_bbox = np.array(all_hand_bbox)

    for k in bad_tracks:
        adj[:, k] = 0
    print(all_hand_bbox.shape)
    for o in range(len(adj)):
        save_file = osp.join(odir, 'VidAnnotations', '%s_%d/hand_inds_side.pkl' % (seqname, o))
        hand_inds = np.where(adj[o] > 0)[0]
        with open(save_file, 'wb') as fp:
            print('save to ', save_file)
            pickle.dump({
                'handside':hand_side, 
                'hand_inds':hand_inds, 
                'hand_box':all_hand_bbox, 
                'bad_hand':bad_tracks
            }, fp)


def build_graph(hand_box, obj_box, contact_box):
    assert len(hand_box) == len(contact_box)
    # NOTE: in the future we should return o2h, not just max(o2h)
    o2h = pairwise_iou(Boxes(torch.FloatTensor(obj_box)), Boxes(torch.FloatTensor(contact_box)))
    o2h = o2h > 0.8
    o2h = o2h.detach().numpy()
    # h_idx = torch.argmax(o2h, -1).detach().numpy()
    # no_exact_match = np.all(contact_box == -1, -1)
    all_hand_zero = np.all(o2h == 0, -1, keepdims=True)
    print(all_hand_zero.shape)
    no_exact_match = np.tile(all_hand_zero, [1, len(hand_box)]) # np.zeros([len(obj_box), len(contact_box)])
    print(no_exact_match.shape)

    o2h_backup = pairwise_dist(obj_box, hand_box)
    # h_idx_backup = np.argmin(o2h_backup, -1)

    # corresp = h_idx * (1 - no_exact_match) + h_idx_backup * no_exact_match
    corresp = o2h * (1 - no_exact_match) + o2h_backup * no_exact_match
    assert len(corresp) == len(obj_box)
    return corresp


def pairwise_dist(box1, box2):
    def box_to_xy(box):
        x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2], -1)
    c1 = box_to_xy(box1)  # N, 2
    c2 = box_to_xy(box2)  # M, 2

    return np.sum((c1[:, None] - c2[None])**2, axis=-1)  # N, M, 2

def vis_hand(seqname):
    image_list = sorted(glob.glob(osp.join(odir, 'JPEGImages', '%s_%d/*.jpg' % (seqname, 0))))
    hand_mask = np.load(osp.join(odir, 'VidAnnotations', '%s_%d/ppl.npy' % (seqname, 0)))

    hand_meta = pickle.load(open(osp.join(odir, 'VidAnnotations', '%s_%d/hand_inds_side.pkl' % (seqname, 0)), 'rb'))
    hand_inds = hand_meta['hand_inds']
    hand_box = hand_meta['hand_box'].astype(np.int)
    color_list = np.random.randint(0, high=255, size=[len(hand_mask), 1, 1, 3])

    image_list = [imageio.imread(fname) for fname in image_list]
    for f in range(len(image_list)):
        p = hand_inds[0]
        image_list[f] = cv2.rectangle(
            image_list[f].copy(), 
            (int(hand_box[p, f, 0]), int(hand_box[p, f, 1])), (int(hand_box[p, f, 2]), int(hand_box[p, f, 3])), 
            (255, 255, 0), 3)
        print(hand_meta['handside'][p])
        for p in range(len(hand_mask)):
            image_list[f] = color_list[p] * hand_mask[p, f][..., None] + (1 - hand_mask[p, f][..., None]) * image_list[f]

    imageio.mimwrite(osp.join(odir, '../vis', '%s.gif' % seqname), image_list)


if __name__ == '__main__':
    seqname = argv[1] if len(argv) >= 2 else '*'
    # vid_list = glob.glob(osp.join(odir, 'JPEGImages', seqname))
    # vid_list = [osp.basename(e) for e in vid_list]
    # for seqname in vid_list:
    find_corsp_hand(seqname)
        # vis_hand(seqname)