from sys import argv
import numpy as np
from glob import glob
import os
import os.path as osp
import cv2
import imageio
from PIL import Image
from nnutils import image_utils

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
    os.makedirs(osp.join(data_dir, 'CropFg', seqname, ), exist_ok=True)
    
    for i, (image, mask) in enumerate(zip(image_list, mask_list)):
        mask = cv2.imread(mask)
        fg = mask_fg(cv2.imread(image), mask)
        bbox = mask_to_bbox(mask)
        bbox= image_utils.square_bbox(bbox, 0.2)
        fg = image_utils.crop_resize(fg, bbox, w, constant_values=255)
        
        cv2.imwrite(osp.join(data_dir, 'CropFg', seqname, osp.basename(image)), fg)
    return 


def mask_fg(image, mask):
    if mask.ndim == 2:
        mask = mask[..., None]
    mask = (mask > 0)
    fg = image * mask + np.array([[[255, 255 ,255]]]) * (1 - mask)
    return fg


def mask_to_bbox(mask):
    indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
    x, y = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
    dx, dy = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
    bbox = np.array([x - dx, y - dy, x+ dx, y + dy])
    return bbox


if __name__ == '__main__':
    crop_fg(argv[1])