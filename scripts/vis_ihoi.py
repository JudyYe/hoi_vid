import argparse
from glob import glob
import os
import os.path as osp
import utils.make_web as web_utils
import pandas

arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
arg_parser.add_argument('--out', default='../output/ihoi_out/', type=str)
args = arg_parser.parse_args()

df = pandas.read_csv('../output/100doh_detectron/by_obj/very_good.csv')
index_list = df['index'].tolist()

data_dir = '../output/100doh_detectron/by_obj/'
ihoi_dir = args.out
html_dir = args.out + '/vis_doh/'
file_list = []

file_list.append(['input', 'original view', 'normalize', 'mesh t0', 'mesh tmid', 'mesh tlast'])
for index in index_list:
    row = []
    # input | output2 | output3 | output4
    # input
    row.append(osp.join(ihoi_dir, index, 'vis/input.gif'))

    row.append(os.path.join(ihoi_dir, index, 'vis/front_view.gif'))
    row.append(os.path.join(ihoi_dir, index, 'vis/hand_view.gif'))

    num_frame = len(glob(os.path.join(ihoi_dir, index, 'meshes/*.obj')))
    row.append(os.path.join(ihoi_dir, index, 'meshes/00000.obj'))
    row.append(os.path.join(ihoi_dir, index, 'meshes/%05d.obj' % (num_frame // 2)))
    row.append(os.path.join(ihoi_dir, index, 'meshes/%05d.obj' % (num_frame - 1)))

    file_list.append(row)

web_utils.run(html_dir, file_list, ('height', 250))