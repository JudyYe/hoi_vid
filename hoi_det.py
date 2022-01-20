# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import shutil

import xml.etree.ElementTree as ET

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import os.path as osp
import json
import sys
import time
import numpy as np
import scipy.io as sio
import tqdm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, pairwise_iou, Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from torchvision import ops
import torch
# from detectron2.projects.PointRend.point_rend import ColorAugSSDTransform, add_pointrend_config
from detectron2.projects import point_rend
import pycocotools.coco as coco
from detectron2.engine import DefaultPredictor

from box2mask import Box2Mask

data_dir = '../data/100doh/'
anno_file = '/glusterfs/yufeiy2/download_data/COCO/annotations/instances_val2017.json'
coco_data = coco.COCO(anno_file)
coco_classes = coco_data.loadCats(coco_data.getCatIds())
coco_classes = [cat['name'] for cat in coco_classes]


def test(seqname):
    # detect the first frame, move object seq to DAVIS/ 
    # rename seq to %05d from 0
    # seqname=sys.argv[1]
    detbase='../detectron2' # sys.argv[2]
    odir='../database/DAVIS/'
    imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
    maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)
    coco_metadata = MetadataCatalog.get("coco_2017_val")
    vis_dir = '/home/yufeiy2/hoi_vid/output/tmp_vis'
    os.makedirs(vis_dir, exist_ok=True)
    cfg = get_cfg()

    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file('%s/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'%(detbase))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.
    cfg.MODEL.WEIGHTS ='https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'

    root = load_anno(seqname)
    predictor = Box2Mask(cfg)

    gt_boxes, is_object = find_gt_boxes(root)
    path = osp.join(data_dir, 'JPEGImages', seqname + '.jpg')
    img = cv2.imread(path)

    predictions = predictor(img, gt_boxes, is_object)

    segs = predictions['instances'].to('cpu')
    masks = segs.pred_masks.cpu().detach().numpy()
    gt_boxes = gt_boxes.tensor.cpu().numpy()
    is_object = is_object.numpy()
    hand_inds = np.where(is_object == 0)[0]
    obj_inds = np.where(is_object == 1)[0]
    hand_mask = ((masks[hand_inds] > 0) * 255).astype(np.uint8)
    obj_mask = ((masks[obj_inds] > 0) * 255).astype(np.uint8)

    # v = Visualizer(img, coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
    # vis = v.draw_instance_predictions(segs)
    # print('write to pred', out_pref)
    # cv2.imwrite(out_pref + '_pred.png', vis.get_image())

    for o in range(len(obj_mask)):
        os.makedirs(maskdir + '_%d' % o, exist_ok=True)
        cv2.imwrite(maskdir + '_%d/%05d.png' % (o, 0), obj_mask[o])
        # cv2.imwrite(maskdir + '_%d/%05d.png' % (o, 0), hand_mask[o])
        sio.savemat(maskdir + '_%d/%05d.mat' % (o, 0), 
            {'hand_mask': hand_mask, 'hand_box': gt_boxes[hand_inds],
            'obj_box': gt_boxes[obj_inds[o]]})

        dst_folder = imgdir + '_%d' % o
        if osp.exists(dst_folder): shutil.rmtree(dst_folder)
        os.makedirs(dst_folder)
        src_folder = osp.join('/home/yufeiy2/hoi_vid/output/100doh_clips/%s/frames/*.jp*' % seqname)
        for i,src_path in enumerate(sorted(glob.glob(src_folder))):
            shutil.copyfile(src_path, osp.join(dst_folder, '%05d.jpg' % i))
        # print(src_folder, dst_folder)
        
        # shutil.copytree(src_folder, dst_folder, )


def main(args, image_set):
    import sys
    seqname=sys.argv[1]
    detbase='../detectron2' # sys.argv[2]
    datadir='../database/DAVIS/JPEGImages/Full-Resolution/%s-tmp/'%seqname
    odir='../database/DAVIS/'
    imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
    maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)
    coco_metadata = MetadataCatalog.get("coco_2017_val")

    import shutil
    if os.path.exists(imgdir): shutil.rmtree(imgdir)
    if os.path.exists(maskdir): shutil.rmtree(maskdir)
    os.mkdir(imgdir)
    os.mkdir(maskdir)

    from detectron2.projects import point_rend
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file('%s/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'%(detbase))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.1
    cfg.MODEL.WEIGHTS ='https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'

    predictor = DefaultPredictor(cfg)

    root = load_anno(seqname)

    counter=0
    for i,path in enumerate(sorted(glob.glob('%s/*'%datadir))):
        print(path)

        img = cv2.imread(path)
        shape = img.shape[:2]
        mask = np.zeros(shape)

        imgt = img
        predictions = predictor(imgt)
        segs = predictions['instances'].to('cpu')

        out_pref = '%s/%05d'%(imgdir,i)

        v = Visualizer(img, coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
        vis = v.draw_instance_predictions(segs)
        cv2.imwrite(out_pref + '_pred.png', vis.get_image())
        
        instances, gt_boxes = filter_iou(root, predictions, 0.25)
        person_mask = merge_sem_mask(predictions)
        # draw gt bbox
        if i == 0:
            canvas = img.copy()
            # bbox: x1y1, x2y2
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(canvas, (int(x1), int(y1)), 
                    (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.imwrite(out_pref + '_gt.png', canvas)

        # output = predictor(im)
        boxes, segms, classes = convert_from_pred_output(instances)
        if len(boxes) == 0:
            print('skip')
            continue

        # class_names = np.asarray(([coco_to_pascal_name(coco_classes[c]) for c in classes]), dtype='object')
        # sio.savemat(out_name + '.mat' , {'masks': segms, 'boxes': boxes, 'classes': class_names, 'person': person_mask})
        # cv2.imwrite(out_name.replace('.png', '_1.png'), im)
        # visualized_output.save(out_name.replace('.png', '_3.png'))

        if i < 100:
            visualized_output = vis_pred(img, instances, cfg)
            visualized_output = vis_gt(visualized_output.get_image(), gt_boxes)
            fname = out_pref + '_vis.png'
            print('save to ', fname)
            cv2.imwrite(fname, visualized_output[:, :, ::-1])
            person_mask = person_mask[:, :, None]
            person = person_mask * visualized_output + (1 - person_mask) * (0.5 + 0.5 * visualized_output)
            fname = out_pref + '_ppl.png'
            cv2.imwrite(fname, person[:, :, ::-1])
        counter += 1


def merge_sem_mask(predictions, th=0.3):
    instances = predictions['instances']
    ps_id = coco_data.getCatIds(['person'])[0] - 1
    mask_list = []
    for n in range(len(instances)):
        if instances.pred_classes[n] == ps_id and instances.scores[n] > th:
            mask = instances.pred_masks[n]
            mask_list.append(mask)
    if len(mask_list) == 0:
        return np.array([-1])
    mask_list = torch.stack(mask_list, dim=0)  # (M, H, W, 1?)
    mask_list = torch.sum(mask_list, dim=0)
    mask_list = mask_list > 0
    mask_list = mask_list.cpu().detach().numpy().astype(np.uint8) * 255
    return mask_list


def vis_gt(image, boxes):
    """
    :param image: (H, W, 3) numpy
    :param boxes: (N, 4)
    :return:
    """
    N = len(boxes)
    boxes = boxes.tensor.cpu().detach().numpy().astype(np.int)
    for n in range(N):
        x1, y1, x2, y2 = boxes[n]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return image


def vis_pred(image, instances, cfg):
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    image = image[:, :, ::-1]

    cpu_device = torch.device("cpu")
    visualizer = Visualizer(image, metadata)
    instances = instances.to(cpu_device)
    vis_output = visualizer.draw_instance_predictions(predictions=instances)
    return vis_output


def find_gt_boxes(root, filter_field=None):  
    """
    filter_field: targetobject / hand
    """
    gt_boxes, is_object = [], []
    for obj in root.findall('object'):
        cls = obj.find('name')
        if filter_field is None or cls.text == filter_field:
            gt_bbox = extract_bbox(obj)
            gt_boxes.append(gt_bbox)
            is_object.append(cls.text == 'targetobject')
    gt_boxes = Boxes(torch.FloatTensor(gt_boxes))
    is_object = torch.LongTensor(is_object)
    return gt_boxes, is_object


def filter_iou(root, prediction, iou_threshold):
    gt_boxes = find_gt_boxes(root, 'targetobject')
    rest_boxes = prediction['instances'].pred_boxes
    gt_boxes = Boxes(torch.FloatTensor(gt_boxes).to(rest_boxes.device))
    N = len(gt_boxes)
    M = len(rest_boxes)

    scores = prediction['instances'].scores
    index = torch.argsort(scores, descending=True)

    index_list = []

    iou = pairwise_iou(gt_boxes, rest_boxes)  # (GT, N)
    iou = iou > iou_threshold
    association = torch.zeros([N]).to(rest_boxes.device).long() - 1
    for n in range(N):
        for j in range(M):
            j = index[j].item()
            if iou[n, j] > 0:
                ps_id = coco_data.getCatIds(['person'])[0]
                if prediction['instances'].pred_classes[j] + 1== ps_id:
                    # index_list.append(j)
                    continue
                # association[n] = j
                index_list.append(j)
                break

    # index = torch.masked_select(index, index >= 0)
    prediction = select_prediction(prediction, index_list)
    return prediction, gt_boxes


def extract_bbox(obj):
    box = obj.find('bndbox')
    x1 = float(box.find('xmin').text)
    x2 = float(box.find('xmax').text)
    y1 = float(box.find('ymin').text)
    y2 = float(box.find('ymax').text)
    return [x1, y1, x2 ,y2]


def select_prediction(output, index):
    instance = Instances(output['instances'].image_size)
    # output['instances'].scores = output['instances'].scores[index]
    # output['instances'].pred_boxes = output['instances'].pred_boxes[index]
    # output['instances'].pred_classes = output['instances'].pred_classes[index]
    # output['instances'].pred_masks = output['instances'].pred_masks[index]
    instance.set('scores', output['instances'].scores[index])
    instance.set('pred_boxes', output['instances'].pred_boxes[index])
    instance.set('pred_classes', output['instances'].pred_classes[index])
    instance.set('pred_masks', output['instances'].pred_masks[index])
    return instance


def convert_from_pred_output(output):
    """
    :param output:
    :return: bbox (N, 5). segms: (N, H, W), classes: (N, nC)
    """
    # output = output['instances']
    score = output.scores.cpu().detach().numpy()
    bbox = output.pred_boxes.tensor.cpu().detach().numpy()
    classes = output.pred_classes.cpu().detach().numpy()
    masks = output.pred_masks.cpu().detach().numpy()

    bbox = np.hstack([bbox, score[..., None]])

    return bbox, masks, classes


def load_index_list(image_set='val'):
    """Images"""
    findex = os.path.join(data_dir, 'ImageSets/Main/%s.txt' % image_set)
    with open(findex) as fp:
        index_list = [line.strip() for line in fp]
    return index_list


def load_anno(index):
    """Images"""
    with open(os.path.join(data_dir, 'Annotations', index + '.xml'), 'r') as fid:
        root = ET.parse(fid).getroot()
    return root


def load_index_vid_list():
    cls2vid_file = osp.join(data_dir, 'VideoMeta', 'class2video.json')
    cls2vid = json.load(open(cls2vid_file))
    clip_list = []
    for cls in cls2vid:
        clip_list.extend(cls2vid[cls])
    return clip_list



if __name__ == '__main__':
    
    # main(None, None)
    vid_list = [line.strip() for line in open('../../code/shots.yaml')]
    for vid in vid_list:
        test(vid)
        # break``