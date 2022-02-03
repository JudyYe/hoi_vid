import numpy as np
from glob import glob
import os
import os.path as osp
import cv2
import imageio
from PIL import Image


data_dir = '../output/100doh_detectron/by_obj/'

def save_fg(seqname):
    image_list = sorted(glob(osp.join(data_dir, 'JPEGImages', seqname, '*.jpg')))
    mask_list = sorted(glob(osp.join(data_dir, 'VidAnnotations', seqname, '*.png')))
    os.makedirs(osp.join(data_dir, 'Foreground', seqname, ), exist_ok=True)

    for i, (image, mask) in enumerate(zip(image_list, mask_list)):
        fg = mask_fg(cv2.imread(image), cv2.imread(mask))
        cv2.imwrite(osp.join(data_dir, 'Foreground', seqname, osp.basename(image)), fg)
    return


def crop_fg(seqname, w=512):
    image_list = sorted(glob(osp.join(data_dir, 'JPEGImages', seqname, '*.jpg')))
    mask_list = sorted(glob(osp.join(data_dir, 'VidAnnotations', seqname, '*.png')))
    os.makedirs(osp.join(data_dir, 'Crop', seqname, ), exist_ok=True)
    
    for i, (image, mask) in enumerate(zip(image_list, mask_list)):
        mask = cv2.imread(mask)
        fg = mask_fg(cv2.imread(image), mask)
        bbox = mask_to_bbox(mask)
        bbox = square_bbox(bbox)
        bbox = pad_bbox(bbox)
        fg = crop_and_resize(fg, bbox, w)
        cv2.imwrite(osp.join(data_dir, 'Crop', seqname, osp.basename(image)), fg)
    return 


def mask_fg(image, mask):
    if mask.ndim == 2:
        mask = mask[..., None]
    fg = image * mask + np.array([[[255, 255 ,255]]]) * mask
    return fg


def mask_to_bbox(mask):
    return 
    
def pad_bbox(bbox):
    return 

def square_bbox(bbox):
    return bbox

def crop_and_resize(fg, bbox, w):
    return fg
