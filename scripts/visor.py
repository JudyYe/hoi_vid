import json
from tqdm import tqdm
import os
import os.path as osp
import argparse

data_dir = '/home/yufeiy/data' # '../data/VISOR'
# data_dir = '/home/yufeiy2/scratch/data/VISOR/'
epic100_url = 'https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P01/rgb_frames/P01_101.tar'
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
    index_list = train_index_list + test_index_list
    index_list = [e.split('.json')[0] for e in index_list]
    os.makedirs(osp.join(data_dir, 'Sets'), exist_ok=True)
    set_file = osp.join(data_dir, 'Sets', 'trainval.txt')
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
    index_list = [e.strip() for e in open(osp.join(data_dir, 'Sets/trainval.txt'))]
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
    
# def pretty_save
def parse_args():
    parser = argparse.ArgumentParser(description='Extract 100DOH')
    parser.add_argument('--dl_anno', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--filter_anno', action='store_true', default=False)
    parser.add_argument('--xclip', action='store_true', default=False)
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
        make_index_list()
        # pretty_save()
    if args.filter_anno:
        filter_annos()
    if args.xclip:
        x_clips_each(args.num)