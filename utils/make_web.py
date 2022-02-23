# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
"""Usage: make_web.run(root, 2D_cell_list, wdith) """
from __future__ import print_function

import glob
import logging
import os
import re
import shutil

import numpy as np
import yaml
from flask_table import Table, Col, create_table
from flask import Markup
import argparse
from jutils import mesh_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=int, default=200, help='')
    args = parser.parse_args()
    return args


def run(html_root, cell_list, size=('width', 600)):
    """
    cell_list: 2D array, each element could be: filepath of vid/image, str
    """
    os.makedirs(html_root, exist_ok=True)
    ncol = len(cell_list[0])
    # title
    TableCls = create_table('TableCls')
    for c in range(ncol):
        TableCls = TableCls.add_column('%d' % c, Col('%d' % c))

    items = []
    for r, row in enumerate(cell_list):
        line = {}
        for c in range(ncol):
            line['%d' % c] = html_add_col_text(row[c], html_root, size, 'r%02dc%02d' % (r, c))
        items.append(line)
    table = TableCls(items)
    html_str = table.__html__()
    with open(os.path.join(html_root, 'index.html'), 'w') as fp:
        # add title 
        fp.write('<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>\n')
        fp.write(html_str)
        print('write to %s.html' % os.path.join(html_root, 'index'))


def html_add_col_text(src_file, vis_dir, size, pref):
    """
    :param col_name: cols to add to the line
    :param line:
    :param file_list: list of file to display in (line, col_name)
    :param vis_dir: copy and cache to vis_dir
    :return:
    """
    key, width = size
    img_temp = '<a href="{0}"><img src="{0}" %s="%d"> </a> <br/> {0} <br/>' % (key, width)
    vid_temp = '<video controls %s="%d"><source src="{0}" type="video/mp4"></video> <br/> {0} <br/>' % (key, width)
    mesh_temp = '<model-viewer src="{0}" style="%s:%d" shadow-intensity="1" camera-controls="" auto-rotate="" ar="" ar-status="not-presenting"></model-viewer>' % (key, width)

    str_temp = '{0}'
    col_text = ''
    if os.path.exists(src_file):
        ext = src_file.split('.')[-1]
        if ext in ['mp4']:
            temp = vid_temp
            dst_file = os.path.join(vis_dir, '%s_%s' % (pref, os.path.basename(src_file)))
            shutil.copyfile(src_file, dst_file)
        elif ext in ['png', 'gif', 'jpg', 'jpeg']:
            temp = img_temp
            dst_file = os.path.join(vis_dir, '%s_%s' % (pref, os.path.basename(src_file)))
            shutil.copyfile(src_file, dst_file)
        elif ext in ['obj', 'ply', 'glb']:
            temp = mesh_temp
            dst_file = os.path.join(vis_dir, '%s_%s' % (pref, os.path.basename(src_file)[:-3] + 'glb'))
            mesh_utils.meshfile_to_glb(src_file, dst_file)
        else:
            dst_file = os.path.join(vis_dir, '%s_%s' % (pref, os.path.basename(src_file)))
            shutil.copyfile(src_file, dst_file)

        col_text += temp.format(os.path.basename(dst_file))
    else:
        col_text += str_temp.format(src_file)
    return Markup(col_text)


if __name__ == '__main__':
    args = parse_args()
    data_dir = '../output/'
    html_root = os.path.join(data_dir, args.exp, 'vis_%s' % args.folder)
    print(html_root)
    os.makedirs(os.path.join(html_root), exist_ok=True)
    run(
        html_root=html_root,
        cell_list=cell_list,
        )
