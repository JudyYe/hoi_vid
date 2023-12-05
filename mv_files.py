import os
import os.path as osp
from glob import glob

src_dir = '/private/home/yufeiy2/scratch/result/wild/'
dst_dir = '/private/home/yufeiy2/scratch/result/wild/'

folder = 'epic_video_clips_fast'

def mv_folder(folder):
    # mv all files under folder to dst_dir
    for src_folder in glob(osp.join(src_dir, folder, '*/*')):
        seqname = osp.basename(src_folder)
        dst_folder = osp.join(dst_dir, seqname)
        # os.makedirs(dst_folder, exist_ok=True)
        cmd = f'mv {src_folder} {dst_folder}'
        print(cmd)
        os.system(cmd)

# mv_folder(folder)
# mv_folder('yt_clips_fast')        

def make_folder(src_dir):
    seq_list = glob(osp.join(src_dir, '*/'))
    for seq_dir in seq_list:
        # make a images/ folder and put all .jpg to that
        new_folder = osp.join(seq_dir, 'images')
        os.makedirs(new_folder, exist_ok=True)

        # for img_path in glob(osp.join(seq_dir, '*.jpg')):
        #     cmd = f'mv {img_path} {new_folder}'
        #     print(cmd)
        #     os.system(cmd)
        cmd = f'mv {new_folder}/bbox.jpg {seq_dir}'
        print(cmd)
        os.system(cmd)

make_folder(osp.join(src_dir, 'raw'))
