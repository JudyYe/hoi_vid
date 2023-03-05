import pandas
from jutils import web_utils
import numpy as np
import cv2
import imageio
import shutil
from collections import defaultdict
from glob import glob
import json
from tqdm import tqdm
import os
import os.path as osp
import argparse

data_dir = '/home/yufeiy/data' # '../data/VISOR'
# data_dir = '/home/yufeiy2/scratch/data/VISOR/'
epic100_url = 'https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P01/rgb_frames/P01_14.tar'
# epic18_url = 'https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P01/P01_11.tar'


cat2id = {'Bowl': 7, 'Bottle': 15, 'Mug': 13, 
       'Knife': 4, 'Kettle': 44}
hand2id = {'right': 301}
rigid_id_list = set(cat2id.values())
id2cat = {v: k for k, v in cat2id.items()}


def make_index_list():
    train_index_list = """P01_01.json, P01_03.json, P01_05.json, P01_07.json, P01_09.json, P01_103.json, P01_104.json, P01_14.json, P02_01.json, P02_03.json, P02_07.json, P02_101.json, P02_102.json, P02_107.json, P02_109.json, P02_121.json, P02_122.json, P02_124.json, P02_128.json, P02_130.json, P02_132.json, P02_135.json, P03_03.json, P03_04.json, P03_05.json, P03_101.json, P03_11.json, P03_112.json, P03_113.json, P03_123.json, P03_13.json, P03_17.json, P03_23.json, P03_24.json, P04_02.json, P04_03.json, P04_04.json, P04_05.json, P04_101.json, P04_109.json, P04_11.json, P04_110.json, P04_114.json, P04_12.json, P04_121.json, P04_21.json, P04_25.json, P04_26.json, P04_33.json, P05_01.json, P05_08.json, P06_01.json, P06_07.json, P06_09.json, P06_101.json, P06_102.json, P06_103.json, P06_107.json, P06_11.json, P06_110.json, P06_12.json, P06_13.json, P06_14.json, P07_08.json, P08_09.json, P08_16.json, P08_21.json, P10_04.json, P11_101.json, P11_102.json, P11_103.json, P11_104.json, P11_105.json, P11_107.json, P11_16.json, P12_02.json, P12_03.json, P12_101.json, P13_10.json, P14_05.json, P15_02.json, P17_01.json, P18_03.json, P18_06.json, P18_07.json, P20_03.json, P22_01.json, P22_07.json, P22_117.json, P23_02.json, P23_05.json, P24_05.json, P24_08.json, P25_107.json, P26_110.json, P27_101.json, P28_06.json, P28_101.json, P28_103.json, P28_109.json, P28_110.json, P28_112.json, P28_113.json, P28_13.json, P28_14.json, P30_05.json, P30_101.json, P30_107.json, P30_111.json, P30_112.json, P32_01.json, P35_105.json, P35_109.json, P37_101.json, P37_103.json"""
    test_index_list = """P01_107.json, P02_02.json, P02_09.json, P02_12.json, P03_10.json, P03_120.json, P03_14.json, P03_22.json, P04_06.json, P04_13.json, P04_24.json, P06_03.json, P06_05.json, P06_10.json, P06_106.json, P06_108.json, P07_101.json, P07_103.json, P07_110.json, P08_17.json, P09_02.json, P09_07.json, P09_103.json, P09_104.json, P09_106.json, P12_04.json, P18_01.json, P18_02.json, P21_01.json, P22_107.json, P24_09.json, P25_09.json, P25_101.json, P26_01.json, P26_02.json, P26_108.json, P27_105.json, P28_05.json, P29_04.json, P30_07.json, P30_110.json, P32_07.json, P37_102.json"""
    train_index_list = train_index_list.split(', ')
    train_index_list = [f'train/{e}' for e in train_index_list]
    test_index_list = test_index_list.split(', ')
    test_index_list = [f'val/{e}' for e in test_index_list]
    # index_list = train_index_list + test_index_list
    index_list = train_index_list
    index_list = [e.split('.json')[0] for e in index_list]
    os.makedirs(osp.join(data_dir, 'Sets'), exist_ok=True)
    set_file = osp.join(data_dir, 'Sets', 'train.txt')
    with open(set_file, 'w') as f:
        for e in index_list:
            f.write(f'{e}\n')
                    
def dl_anno_list():
    train_index_list = """P01_01.json, P01_03.json, P01_05.json, P01_07.json, P01_09.json, P01_103.json, P01_104.json, P01_14.json, P02_01.json, P02_03.json, P02_07.json, P02_101.json, P02_102.json, P02_107.json, P02_109.json, P02_121.json, P02_122.json, P02_124.json, P02_128.json, P02_130.json, P02_132.json, P02_135.json, P03_03.json, P03_04.json, P03_05.json, P03_101.json, P03_11.json, P03_112.json, P03_113.json, P03_123.json, P03_13.json, P03_17.json, P03_23.json, P03_24.json, P04_02.json, P04_03.json, P04_04.json, P04_05.json, P04_101.json, P04_109.json, P04_11.json, P04_110.json, P04_114.json, P04_12.json, P04_121.json, P04_21.json, P04_25.json, P04_26.json, P04_33.json, P05_01.json, P05_08.json, P06_01.json, P06_07.json, P06_09.json, P06_101.json, P06_102.json, P06_103.json, P06_107.json, P06_11.json, P06_110.json, P06_12.json, P06_13.json, P06_14.json, P07_08.json, P08_09.json, P08_16.json, P08_21.json, P10_04.json, P11_101.json, P11_102.json, P11_103.json, P11_104.json, P11_105.json, P11_107.json, P11_16.json, P12_02.json, P12_03.json, P12_101.json, P13_10.json, P14_05.json, P15_02.json, P17_01.json, P18_03.json, P18_06.json, P18_07.json, P20_03.json, P22_01.json, P22_07.json, P22_117.json, P23_02.json, P23_05.json, P24_05.json, P24_08.json, P25_107.json, P26_110.json, P27_101.json, P28_06.json, P28_101.json, P28_103.json, P28_109.json, P28_110.json, P28_112.json, P28_113.json, P28_13.json, P28_14.json, P30_05.json, P30_101.json, P30_107.json, P30_111.json, P30_112.json, P32_01.json, P35_105.json, P35_109.json, P37_101.json, P37_103.json"""
    test_index_list = """P01_107.json, P02_02.json, P02_09.json, P02_12.json, P03_10.json, P03_120.json, P03_14.json, P03_22.json, P04_06.json, P04_13.json, P04_24.json, P06_03.json, P06_05.json, P06_10.json, P06_106.json, P06_108.json, P07_101.json, P07_103.json, P07_110.json, P08_17.json, P09_02.json, P09_07.json, P09_103.json, P09_104.json, P09_106.json, P12_04.json, P18_01.json, P18_02.json, P21_01.json, P22_107.json, P24_09.json, P25_09.json, P25_101.json, P26_01.json, P26_02.json, P26_108.json, P27_105.json, P28_05.json, P29_04.json, P30_07.json, P30_110.json, P32_07.json, P37_102.json"""
    train_index_list = train_index_list.split(', ')
    train_index_list = [f'train/{e}' for e in train_index_list]
    test_index_list = test_index_list.split(', ')
    test_index_list = [f'val/{e}' for e in test_index_list]
    index_list = train_index_list + test_index_list
    index_list = [e.split('.json')[0] for e in index_list]

    url_dir = "https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/annotations/{}"
    for index in tqdm(index_list):
        url = url_dir.format(index)
        split = index.split('/')[0]
        save_dir  = f'{data_dir}/GroundTruth-SparseAnnotations/annotations/{split}'
        os.makedirs(save_dir, exist_ok=True)
        cmd = f"wget {url} -P {save_dir}"
        print(cmd)
        os.system(cmd)


def get_clips(anno_file):
    anno_vid = json.load(open(anno_file))['video_annotations']
    clips = []
    for key_frame in anno_vid:
        anno_dict = {}
        obj_id, hand_id = None, None
        for anno in key_frame['annotations']:
            anno_dict[anno['id']] = anno
            # find right hand in contact
            if anno['class_id'] == 301 \
                and anno['in_contact_object'] != 'hand-not-in-contact'\
                and anno['in_contact_object'] != 'inconclusive':
                obj_id = anno['in_contact_object']
                hand_id = anno['id']
        if obj_id is None or hand_id is None or obj_id not in anno_dict:
            continue
        if not anno_dict[obj_id]['class_id'] in rigid_id_list:
            continue
        
        clips.append({
            'hand': anno_dict[hand_id], 'obj': anno_dict[obj_id], 
            'frame': key_frame['image'],
            })
    print(len(clips))
    return clips


def filter_annos():
    # find frames that the active object is in rigid_id_list
    index_list = [e.strip() for e in open(osp.join(data_dir, 'Sets/train.txt'))]
    all_clips = []
    cnt = 0
    for index in tqdm(index_list):
        anno_file = osp.join(data_dir, 'GroundTruth-SparseAnnotations/annotations', index) + '.json'
        clips = get_clips(anno_file)        
        if len(clips) > 0:
            cnt += 1
        all_clips.extend(clips)
    print('total clips: ', len(all_clips), cnt/len(index_list) * 100)
    with open(osp.join(data_dir, 'Sets/hoi_clips.json'), 'w') as f:
        json.dump(all_clips, f, indent=4)


def sort_by_obj_class_num(all_clips):
    # sort by diversity of object class in each videos
    num_cats_in_video = {}
    for clip in all_clips:
        video = clip['frame']['video']
        obj_cat = clip['obj']['class_id']
        if video not in num_cats_in_video:
            num_cats_in_video[video] = set()
        num_cats_in_video[video].add(obj_cat)
    for clip in all_clips:
        video = clip['frame']['video']
        clip['num_cats'] = len(num_cats_in_video[video])
    all_clips = sorted(all_clips, key=lambda x: x['num_cats'], reverse=True)
    
    return all_clips


def x_clips_each(num=5):
    # get num clips for each category while minimize the videos to download
    all_clips = json.load(open(osp.join(data_dir, 'Sets/hoi_clips.json')))
    all_clips = sort_by_obj_class_num(all_clips)

    cat_count = {k: [] for k in rigid_id_list}
    video_count = {}

    mini_clip_list = []
    for clip in all_clips:
        obj_cat = clip['obj']['class_id']
        if len(cat_count[obj_cat]) >= num:
            continue
        video_count[clip['frame']['video']] = 0
        cat_count[obj_cat].append(clip)

    print(len(video_count), 'vids to download')
    print(video_count)
    # change key from int to cat name by id2cat
    cat_count = {id2cat[k]: v for k, v in cat_count.items()}
    with open(osp.join(data_dir, f'Sets/hoi_clips_minix{num}.json'), 'w') as f:
        json.dump(cat_count, f, indent=4)

    # save video_count key by lines 
    with open(osp.join(data_dir, f'Sets/vids_minix{num}.txt'), 'w') as f:
        for k in video_count:
            f.write(k + '\n')
    
def mv_frame():
    with open(osp.join(data_dir, f'Sets/vids_minix10.txt')) as f:
        vid_list = [e.strip() for e in f.readlines()]
    for vid in vid_list:
        sub = vid.split('_')[0]
        vid_dir = osp.join(data_dir, 'GroundTruth-SparseAnnotations/rgb_frames/train/', sub, vid)
        fname_list = glob(f'{vid_dir}*.jpg')
        if len(fname_list) == 0:
            print(vid, 'nah!')
            continue
        cmd = f'cp {vid_dir}*.jpg {vid_dir}'
        os.makedirs(vid_dir, exist_ok=True)
        print(cmd)
        os.system(cmd)

def unzip_dense_anno():
    vid_list = [e.strip() for e in open(osp.join(data_dir, 'Sets/vids_minix10.txt'))]
    for vid in vid_list:
        # unzip A to dir B
        cmd = f'unzip {data_dir}/Interpolations-DenseAnnotations/train/{vid}_interpolations.zip' +  \
            f' -d {data_dir}/Interpolations-DenseAnnotations/train/'
        os.system(cmd)

def link_vid():
    vid_list = [e.strip() for e in open(osp.join(data_dir, 'Sets/vids_minix10.txt'))]
    for vid in vid_list:
        sub = vid.split('_')[0]
        vid_dir = '/home/yufeiy/data/VISOR/EPIC-KITCHENS'
        dst = f'{vid_dir}/{vid}.MP4'
        cmd = f'ln -s {vid_dir}/{sub}/videos/{vid}.MP4 {dst}'
        if osp.exists(dst):
            continue
        print(cmd)
        os.system(cmd)

def check_one_dense():
    index = 'P03_04'
    dense_file = osp.join(data_dir, 'Interpolations-DenseAnnotations/train', index) + '_interpolations.json'
    with open(dense_file) as f:
        dense_anno = json.load(f)
    # save  dnse_file to vis dir with indent 4
    with open(osp.join(data_dir, 'VISOR/vis', index) + '_dense.json', 'w') as f:
        json.dump(dense_anno, f, indent=4)
        print('saved to ', f.name)


def find_clip():
    cat_count = json.load(open(osp.join(data_dir, 'Sets/hoi_clips_minix10.json')))
    # preload dense anno
    anno_cache = {}
    for cat in tqdm(cat_count):
        vid = cat_count[cat][0]['frame']['video']
        dense_anno = json.load(open(osp.join(data_dir, 'Interpolations-DenseAnnotations/train', vid) + '_interpolations.json'))
        dense_anno = dense_anno['video_annotations']
        anno_cache[vid] = dense_anno
        break
    # import pdb; pdb.set_trace()
    save_list = []
    record = {}
    for cat in tqdm(cat_count, desc='cat'):
        print(cat, len(cat_count[cat]))
        for clip in tqdm(cat_count[cat], desc='clip'):
            vid = clip['frame']['video']
            if vid not in anno_cache:
                dense_anno = json.load(open(osp.join(data_dir, 'Interpolations-DenseAnnotations/train', vid) + '_interpolations.json'))
                dense_anno = dense_anno['video_annotations']
                anno_cache[vid] = dense_anno
            dense_anno = anno_cache[vid]
            frame2ind = {e['image']['name'].split('.')[0]: ind for ind, e in enumerate(dense_anno)}

            frame_index = clip['frame']['name'].split('.')[0]
            # find the frame in dense_anno by query ['image']['name']
            try:
                ind_t = frame2ind[frame_index]
            except KeyError: 
                print('key error', frame_index)
                # find the closest frame
                frame_close = min(frame2ind.keys(), key=lambda x: abs(int(x.split('frame_')[-1]) - int(frame_index.split('frame_')[-1])))
                frame_index = frame_close
                ind_t = frame2ind[frame_close]
                # continue
            # pdb.set_trace()
            obj_id = clip['obj']['class_id']
            
            hand_id = clip['hand']['class_id']

            t_start = max(0, ind_t - 30)
            t_end = min(ind_t + 30, len(dense_anno)-1)
            t_list = range(t_start, ind_t)
            for t in t_list:
                id_list = [e['class_id'] for e in dense_anno[t]['annotations']]
                if hand_id not in id_list:
                    t_start += 1
                break
            t_list = range(t_end, ind_t, -1)
            for t in t_list:
                id_list = [e['class_id'] for e in dense_anno[t]['annotations']]
                if hand_id not in id_list:
                    t_end -= 1
                break
            

            for t in range(t_start, t_end):
                if hand_id in id_list:
                    continue
                else:
                    print( 'no hand')
                    break
            frames_to_save = {
                'key_frame': frame_index,
                'cat': cat,
                'frames': [],
            }

            for t in range(t_start, t_end):
                index = dense_anno[t]['image']['name'].split('.')[0]
                frames_to_save['frames'].append(index)
            save_list.append(frames_to_save)
        print('temp save')
        with open(osp.join(data_dir, 'Sets/hoi_clips_minix10_tt.json'), 'w') as f:
            json.dump(save_list, f, indent=4)
    print(len(save_list))
    # for each in save_list:
        # print(each['cat'], len(each['frames']))
    return 

def cp_clip():
    """copy full_frame clip (H/O mask and hand) to new folder"""
    clip_list = json.load(open(osp.join(data_dir, 'Sets/hoi_clips_minix10_tt.json')))
    dense_rgb_dir = '/home/yufeiy/data/VISOR/out'
    dense_anno_dir = '/home/yufeiy/data/VISOR/mask_dense'
    clip_dir = '/home/yufeiy/data/VISOR/clips'
    
    cat_id = defaultdict(int)
    for clip in tqdm(clip_list, desc='clip'):
        cat = clip['cat']
        key_frame = clip['key_frame']
        frames = clip['frames']
        cat_id[cat] = cat_id[cat] + 1
        save_dir = osp.join(clip_dir, cat + f'_{cat_id[cat]:02d}')
        seq = key_frame.split('_frame_')[0]
        # if seq == 'P03_04':
        #     continue
        for frame in frames:
            basename = osp.join(seq, frame)
            # copy rgb
            rgb_path = osp.join(dense_rgb_dir,  basename + '.jpg')
            new_rgb_path = osp.join(save_dir, 'frame_image', frame + '.jpg')
            anno_path = osp.join(dense_anno_dir, basename + '.png')
            new_anno_path = osp.join(save_dir, 'frame_mask', frame + '.png')
            if not osp.exists(rgb_path):
                print('not exist', rgb_path)
                continue
            if not osp.exists(anno_path):
                # print('not exist', anno_path)
                continue
            os.makedirs(osp.dirname(new_rgb_path), exist_ok=True)
            # copy anno
            os.makedirs(osp.dirname(new_anno_path), exist_ok=True)
            shutil.copy(rgb_path, new_rgb_path)
            shutil.copy(anno_path, new_anno_path)

        # save frame info
        if osp.exists(save_dir):
            json.dump(clip, open(osp.join(save_dir, 'frame_info.json'), 'w'), indent=4)
        else:
            print('seq whole seq', seq, save_dir)
        # break

def vis_clip():
    seq_list = glob(osp.join(data_dir, 'VISOR/clips/*/frame_info.json'))
    vis_dir = osp.join(data_dir, 'VISOR/vis')
    seq_list = [osp.dirname(e) for e in seq_list]

    for seq_dir in seq_list:
        seq = osp.basename(seq_dir)
        # overlay mask on image and save
        image_list = []
        img_list = sorted(glob(osp.join(seq_dir, 'frame_image/*.jpg')))
        mask_list = sorted(glob(osp.join(seq_dir, 'frame_mask/*.png')))
        for img_path, mask_path in zip(img_list, mask_list):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            mask = mask * 0.5
            img = img * 0.5 + mask
            cv2.resize(img, (320, 240))
            image_list.append(img.astype(np.uint8))
        
        imageio.mimsave(osp.join(vis_dir, f'{seq}.gif'), image_list, fps=10)

def web_clip():
    # list all gif in vis_dir
    vis_dir = osp.join(data_dir, 'VISOR/vis')
    clip_dir = osp.join(data_dir, 'VISOR/clips')
    gif_list = sorted(glob(osp.join(vis_dir, '*.gif')))
    cell_list = []

    for gif_path in gif_list:
        line = []
        seq = osp.basename(gif_path).split('.')[0]
        try:
            frame_meta = json.load(open(osp.join(clip_dir, seq, 'frame_info.json')))
        except FileNotFoundError:
            print('no frame info', seq)
            continue

        mask_file_list = sorted(glob(osp.join(clip_dir, seq, 'frame_mask/*.png')))
        mask_list = [imageio.imread(e) for e in mask_file_list]
        fname = osp.join(vis_dir, f'{seq}_mask.gif')
        imageio.mimsave(fname, mask_list, fps=10)
        line.append(fname)

        gif_name = osp.basename(gif_path)
        line.append(osp.basename(gif_name.split('.')[0]))
        line.append(gif_path)
        key_frame = frame_meta['key_frame']
        t_ind = frame_meta['key_frame'].split('_frame_')[1]
        t_start = frame_meta['frames'][0].split('_frame_')[1]
        t_end = frame_meta['frames'][-1].split('_frame_')[1]
        line.extend([key_frame])
        cell_list.append(line)
    web_utils.run(osp.join(vis_dir, 'vis_clip'), cell_list, width=400)


def web_better_clip():
    # list all gif in vis_dir
    vis_dir = osp.join(data_dir, 'VISOR/vis')
    clip_dir = osp.join(data_dir, 'VISOR/clips')
    gif_list = sorted(glob(osp.join(clip_dir, '*_???')))
    cell_list = []

    for gif_path in gif_list:
        line = []
        seq = osp.basename(gif_path).split('.')[0]
        hand_mask_list = sorted(glob(osp.join(gif_path, 'frame_hand_mask/*.png')))
        obj_mask_list = sorted(glob(osp.join(gif_path, 'frame_obj_mask/*.png')))
        mask_list = sorted(glob(osp.join(gif_path, 'frame_mask/*.png')))
        image_list = sorted(glob(osp.join(gif_path, 'frame_image/*.jpg')))

        # overlay hand_mask(green) and obj_mask(red) on top of image
        image_list = []
        for img_path, hand_path, obj_path, mask_path in zip(image_list, hand_mask_list, obj_mask_list, mask_list):
            img = imageio.imread(img_path)
            hand_mask = imageio.imread(hand_path)
            obj_mask = imageio.imread(obj_path)
            mask = imageio.imread(mask_path)
            print(hand_mask.max())
            hand_mask = hand_mask > 122 
            obj_mask = obj_mask > 122
            # hand_mask = hand_mask[..., 0]
            # obj_mask = obj_mask[..., 0]
            # mask = mask * 0.5
            mask = np.stack([obj_mask, hand_mask, np.zeros_like(obj_mask)], -1)
            img = img * 0.5 + mask
            # img = img * 0.5 + hand_mask + obj_mask + mask
            cv2.resize(img, (320, 240))
            image_list.append(img.astype(np.uint8))
        gif_path = osp.join(vis_dir, f'{seq}.gif')
        imageio.mimsave(gif_path, image_list, fps=10)
        gif_name = osp.basename(gif_path)
        line.append(osp.basename(gif_name.split('.')[0]))
        line.append(gif_path)
        # write hand_mask 
        hand_img_list = [imageio.imread(e) for e in hand_mask_list]
        imageio.mimsave(osp.join(vis_dir, f'{seq}_hand.gif'), hand_img_list, fps=10)
        line.append(osp.join(vis_dir, f'{seq}_hand.gif'))
        # write obj_mask
        obj_img_list = [imageio.imread(e) for e in obj_mask_list]
        imageio.mimsave(osp.join(vis_dir, f'{seq}_obj.gif'), obj_img_list, fps=10)
        line.append(osp.join(vis_dir, f'{seq}_obj.gif'))
        # write mask
        mask_img_list = [imageio.imread(e) for e in mask_list]
        imageio.mimsave(osp.join(vis_dir, f'{seq}_mask.gif'), mask_img_list, fps=10)
        line.append(osp.join(vis_dir, f'{seq}_mask.gif'))


        cell_list.append(line)
    web_utils.run(osp.join(vis_dir, 'vis'), cell_list, width=400)


def better_clip():
    dense_rgb_dir = '/home/yufeiy/data/VISOR/out'
    dense_anno_dir = '/home/yufeiy/data/VISOR/mask_dense'
    clip_dir = '/home/yufeiy/data/VISOR/clips'
    
    few_list = ['Kettle_102', 'Bowl_102', 'Bottle_101', 'Bottle_102', "Kettle_101"]
    # few_list = ['Kettle_101']#, 'Bowl_102', 'Bottle_101', 'Bottle_102', "Kettle_101"]
    with open(osp.join('visor_util/visor_shot.yaml')) as f:
        line_list = [e.strip() for e in f.readlines()]
    for line in line_list:
        print(line)
        _, save_index, start_frame, end_frame, side = line.split(',')
        if few_list is not None and save_index not in few_list:
            continue
        start_ind = int(start_frame.split('_frame_')[1].split('.')[0])
        end_ind = int(end_frame.split('_frame_')[1].split('.')[0])
        cat = save_index.split('_')[0]
        seq = start_frame.split('_frame_')[0]
        data_map = pandas.read_csv(osp.join(dense_anno_dir, seq, 'data_mapping.csv'))
        # build dict from video_id to object_name
        video_id_to_obj = {}
        obj_to_video_id = {}
        for _, row in data_map.iterrows():
            video_id_to_obj[row['video_id']] = row['object_name']
            obj_to_video_id[row['object_name']] = row['video_id']
        for t in range(start_ind, end_ind):
    
            frame = f'{start_frame.split("_frame_")[0]}_frame_{t:010d}'
            basename = osp.join(start_frame.split('_frame_')[0], frame)
            
            # copy rgb
            rgb_path = osp.join(dense_rgb_dir,  basename + '.jpg')
            new_rgb_path = osp.join(clip_dir, save_index, 'frame_image', frame + '.jpg')
            anno_path = osp.join(dense_anno_dir, basename + '.png')
            new_hand_path = osp.join(clip_dir, save_index, 'frame_hand_mask', frame + '.png')
            new_obj_path = osp.join(clip_dir, save_index, 'frame_obj_mask', frame + '.png')
            new_anno_path = osp.join(clip_dir, save_index, 'frame_mask', frame + '.png')
            if not osp.exists(rgb_path):
                print('not exist', rgb_path)
                continue
            if not osp.exists(anno_path):
                # print('not exist', anno_path)
                continue
            os.makedirs(osp.dirname(new_rgb_path), exist_ok=True)
            os.makedirs(osp.dirname(new_anno_path), exist_ok=True)
            os.makedirs(osp.dirname(new_hand_path), exist_ok=True)
            os.makedirs(osp.dirname(new_obj_path), exist_ok=True)
            hand_id = video_id_to_obj[side + ' hand']
            obj_id = video_id_to_obj[cat.lower()]
            # shutil.copy(anno_path, new_anno_path)
            mask = np.load(anno_path+'.npz')['arr_0']
            hand_mask = get_mask(mask, hand_id)
            obj_mask = get_mask(mask, obj_id)
            # hand_mask = (mask[..., 1] == hand_id).astype(np.uint8) * 255
            # obj_mask = (mask[..., 1] == obj_id).astype(np.uint8) * 255
            if side == 'left':
                hand_mask = cv2.flip(hand_mask, 1)
                obj_mask = cv2.flip(obj_mask, 1)
                mask = cv2.flip(mask, 1)
            if hand_mask.max() == 0: 
                print(hand_mask.max(), frame)

            img = cv2.imread(rgb_path)
            if side == 'left':
                img = cv2.flip(img, 1)
            
            print('image', img.shape)
            cv2.imwrite(new_anno_path, pad_to_square(mask))
            cv2.imwrite(new_hand_path, pad_to_square(hand_mask))
            cv2.imwrite(new_obj_path, pad_to_square(obj_mask))
            cv2.imwrite(new_rgb_path, pad_to_square(img))

        text = cat
        with open(osp.join(clip_dir, save_index, 'text.txt'), 'w') as f:
            f.write(text)
    

def get_mask(mask, ind):
    print(mask.shape)
    return (mask == ind).astype(np.uint8) * 255
    # davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
    # davis_palette[:104, :] = [[0,0,0],[200, 0, 0], [0, 200, 0],[200, 128, 0], [0, 0, 200], [200, 0, 200], [0, 200, 200], [200, 200, 200],[252,93,82], [160,121,99], [164,188,119], [0,60,29], [75,237,255], [148,169,183], [96,74,207], [255,186,255], [255,218,231], [136,30,23], [231,181,131], [219,226,216], [0,196,107], [0,107,119], [0,125,227], [153,134,227], [91,0,56], [86,0,7], [246,207,195], [87,51,0], [125,131,122], [187,237,218], [46,57,59], [164,191,255], [37,29,57], [144,53,104], [79,53,54], [255,163,128], [255,233,180], [68,100,62], [0,231,199], [0,170,233], [0,20,103], [195,181,219], [148,122,135], [200,128,129], [46,20,10], [86,78,24], [180,255,188], [0,36,33], [0,101,139], [50,60,111], [188,81,205], [168,9,70], [167,91,59], [35,32,0], [0,124,28], [0,156,145], [0,36,57], [0,0,152], [89,12,97], [249,145,183],[255,153,170], [255,153,229], [184,143,204], [208,204,255], [11,0,128], [69,149,230], [82,204,194], [77,255,136], [6,26,0], [92,102,41], [102,85,61], [76,45,0], [229,69,69], [127,38,53], [128,51,108], [41,20,51], [25,16,3], [102,71,71], [77,54,71], [143,122,153], [42,41,51], [4,0,51], [31,54,77], [204,255,251], [51,128,77], [61,153,31], [194,204,143], [255,234,204], [204,119,0], [204,102,102],[64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],[64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],[0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],[0, 64, 128], [128, 64, 128]] # first 90 for the regular colors and the last 14 for objects having more than one segment

    # target_color = davis_palette[ind]
    # return (mask == target_color).all(axis=2).astype(np.uint8) * 255



# pad to square
def pad_to_square(image):
    h, w = image.shape[:2]
    if h > w:
        pad = (h - w) // 2
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = (w - h) // 2
        image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return image
def parse_args():
    parser = argparse.ArgumentParser(description='Extract 100DOH')
    parser.add_argument('--dl_anno', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--filter_anno', action='store_true', default=False)
    parser.add_argument('--xclip', action='store_true', default=False)
    parser.add_argument('--mv_frame', action='store_true', default=False)
    parser.add_argument('--unzip_dense_anno', action='store_true', default=False)
    parser.add_argument('--link_vid', action='store_true', default=False)
    parser.add_argument('--find_clip', action='store_true', default=False)
    parser.add_argument('--cp_clip', action='store_true', default=False)
    parser.add_argument('--vis_clip', action='store_true', default=False)
    parser.add_argument('--web_clip', action='store_true', default=False)
    parser.add_argument('--better_clip', action='store_true', default=False)

    parser.add_argument('--num', default=5, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.dl_anno:
        dl_anno_list()
    if args.debug:
        # anno_file = '/compute/grogu-2-9/yufeiy2/data/VISOR/GroundTruth-SparseAnnotations/annotations/val/P18_02.json'
        # get_clips(anno_file)
        # make_index_list()
        # pretty_save()
        check_one_dense()
    if args.filter_anno:
        filter_annos()
    if args.xclip:
        x_clips_each(args.num)
    if args.mv_frame:
        mv_frame()
    if args.unzip_dense_anno:
        unzip_dense_anno()
    if args.link_vid:
        link_vid()

    if args.find_clip:
        find_clip()

    if args.cp_clip:
        cp_clip()

    if args.vis_clip:
        vis_clip()

    if args.better_clip:
        better_clip()

    if args.web_clip:
        # web_clip()
        web_better_clip()
    
    # if args.just_few:
    #     just_few()
