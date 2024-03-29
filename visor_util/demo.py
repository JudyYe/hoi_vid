#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:04:58 2022

@author: Ahmad Darkhalil
"""
from vis import *
import os

# json_files_path = '/home/yufeiy/data/Interpolations-DenseAnnotations/train' # 
json_files_path = '/home/yufeiy/data/GroundTruth-SparseAnnotations/annotations/train' 
# '/home/yufeiy/data/GroundTruth-SparseAnnotations/annotations/train' # ../json_files'
output_directory = '/home/yufeiy/data/VISOR/vis/' # '../outputs'
output_resolution= (854,480)
is_overlay=False
rgb_frames = '/home/yufeiy/data/GroundTruth-SparseAnnotations/rgb_frames/train/P01'
generate_video=True
seq = 'P01_14'
folder_of_jsons_to_masks(json_files_path, output_directory,is_overlay=is_overlay,rgb_frames=rgb_frames,output_resolution=output_resolution,generate_video=generate_video, query=seq)
