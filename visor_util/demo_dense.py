#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:04:58 2022

@author: Ahmad Darkhalil
"""
from vis import *
import os

from sys import argv
# json_files_path = '/home/yufeiy/data/Interpolations-DenseAnnotations/train' # 
json_files_path = '/home/yufeiy/data/Interpolations-DenseAnnotations/train/' 
# '/home/yufeiy/data/GroundTruth-SparseAnnotations/annotations/train' # ../json_files'
output_directory = '/home/yufeiy/data/VISOR/mask_dense' # '../outputs'
output_resolution= (854,480)
is_overlay=False
rgb_frames = '/home/yufeiy/data/VISOR/out'
generate_video=True
seq = argv[1] # 'P03_04_*'
print(seq)
folder_of_jsons_to_masks(json_files_path, output_directory,is_overlay=is_overlay,rgb_frames=rgb_frames,output_resolution=output_resolution,generate_video=generate_video, query=seq)
