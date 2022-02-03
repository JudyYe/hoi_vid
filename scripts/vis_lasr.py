from glob import glob
import os
import utils.make_web as web_utils
from utils import ffmpeg_utils
import pandas


df = pandas.read_csv('../output/100doh_detectron/by_obj/very_good.csv')
index_list = df['index'].tolist()

html_dir = '../output/lasr_output/vis_doh/'
data_dir = '../output/100doh_detectron/by_obj/'
lasr_dir = '../output/lasr_output/tmp/'
file_list = []

file_list.append(['input', 'mask', 'original view', 'shape', 'bones'])
for index in index_list:
    row = []
    # input | mask | output2 | output3 | output4
    # input
    dst_file = os.path.join(html_dir, index + '_inp')
    ffmpeg_utils.write_mp4(sorted(glob(os.path.join(data_dir, 'JPEGImages', index, '*.jpg'))), dst_file, clear=False)
    row.append(dst_file + '.mp4')

    # mask
    dst_file = os.path.join(html_dir, index + '_mask')
    ffmpeg_utils.write_mp4(sorted(glob(os.path.join(data_dir, 'VidAnnotations', index, '*.png'))), dst_file, clear=False)
    row.append(dst_file + '.mp4')

    # output2
    row.append(os.path.join(lasr_dir, index + '2.gif'))
    # output3
    row.append(os.path.join(lasr_dir, index + '3.gif'))
    # output4
    row.append(os.path.join(lasr_dir, index + '4.gif'))

    file_list.append(row)

web_utils.run(html_dir, file_list, 600)