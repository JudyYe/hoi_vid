import xml.etree.ElementTree as ET
import pandas
import numpy as np
import os
import os.path as osp
from pytube import YouTube
from tqdm import tqdm
import glob
from ffmpeg_utils import arithmeticEval, cvt_frame_num, write_mp4, extract_frame

data_dir = '../data/100doh/'
vid_dir = '../data/100doh/Videos'
total_num = 100
rgb_dir = '../output/extracted_100doh/'
shot_dir = '../output/100doh_shots/'
clip_dir = '../output/100doh_clips/'
# downlaod random videos and extract them
# see their segments
# visualize annotation if possible 

# ffmpeg -i test.mp4 -vf fps=0.5 -qscale:v 2 test/frame%06d.jpg
def download_videos(frame_index_list=None):
    meta_file = os.path.join(data_dir, 'VideoMeta', 'meta_100DOH_video.txt')
    df = pandas.read_csv(meta_file, delim_whitespace=True, index_col=False)
    np.random.seed(123)
    df = df.sample(frac=1)

    cnt = 0
    for i, (index, data) in enumerate(df.iterrows()):
        vid = data['video_id']
        genre = data['genre']
        url = data['link']

        print('grab vid', vid, genre)

        vid_name = '%s_v_%s' % (genre, vid)
        ret = download_video(url, vid_name, save_dir=vid_dir)
        if not ret:
            continue
        
        cnt += 1
        if cnt >= total_num:
            break

# youtube-dl [OPTIONS] URL [URL...]
def download_video(url=None, vid_name='test', save_dir=''):
    os.makedirs(save_dir, exist_ok=True)
    if osp.exists(osp.join(save_dir, vid_name + '.mp4')):
        return True
    try:
        yt = YouTube(url)
    except:
        print(url)
        print("Connection Error")  # to handle exception
        return False
    print('downloading video for ', url)
    try:
        video = yt.streams.get_highest_resolution()
        # video = yt.streams.get_by_resolution(reso)
        video.download(save_dir, filename=vid_name + '.mp4')
    except:
        print('Cannot download', url)
        return False
    print('Task Completed!')
    return True


def split_shot():
    return

def load_shot_info(vid):
    label = vid.split('_')[0]
    shot_list = [line.strip() for line in open(osp.join(data_dir, 'shot', label, vid + '_shot_info.txt'))]
    shot_list = [[int(f) for f in e.split()] for e in shot_list]
    shot_list = np.array(shot_list)
    return shot_list

def extract_shot():
    video_list = glob.glob(osp.join(vid_dir, '*.mp4'))
    vid_id_list = [osp.basename(e).split('.')[0] for e in video_list]

    for vid in vid_id_list:
        shot_list = load_shot_info(vid)
        frame_list = np.array(sorted(glob.glob(osp.join(dst_dir, vid, '*.jpg'))))
        num_shot = shot_list[-1, 1]
        for ss in range(num_shot):
            idx = np.where(shot_list[:, 1] == ss)[0]
            print(idx)

            time = len(idx) // 30 
            shot_type = shot_list[idx[0], -1]
            shot_file = osp.join(shot_dir, vid, '%d_%d_%d' % (ss, shot_type, time))
            write_mp4(frame_list[idx], shot_file)
    
def find_min_max_shot(start_frame_orig, shot_info):
    frame_inds = start_frame_orig - 1 
    shot_inds = shot_info[frame_inds, 1]
    indices = np.where(shot_info[:, 1] == shot_inds)[0]
    min_idx = np.min(indices)
    max_idx = np.max(indices)
    return min_idx + 1, max_idx + 1
    
def extract_key_frames():
    df = pandas.read_csv(osp.join(data_dir, 'VideoMeta', 'meta_100K_frame.txt'), delim_whitespace=True, index_col=False)
    video_list = glob.glob(osp.join(vid_dir, '*.mp4'))
    vid_id_list = [osp.basename(e).split('.')[0] for e in video_list]
    for vid in vid_id_list:
        if not has_contact(vid):
            print('skip', vid)
            continue
        vid_id = vid.split('_v_')[-1]
        frame_list = df[df['video_id'] == vid_id]

        shot_info = load_shot_info(vid)
        print(len(frame_list))
        for i, (index, data) in enumerate(frame_list.iterrows()):
            frame_index = int(data['frame_index'] )
            fps = data['frame_rate']
            print(vid, frame_index, fps)
            dst_file = osp.join(clip_dir, '%s_frame%06d' % (vid, frame_index), 'clip')
            # if osp.exists(dst_file + '.mp4'):
            #     print('Exist file pass: ', dst_file)
            #     continue

            start_frame_orig = cvt_frame_num(frame_index, 0.5, fps)
            min_frame, max_frame = find_min_max_shot(start_frame_orig, shot_info)
            print(min_frame, max_frame, start_frame_orig)
            time_len = min((max_frame - start_frame_orig) / arithmeticEval(fps), 20)

            extract_frame(osp.join(vid_dir, vid), 
                osp.join(clip_dir, '%s_frame%06d/frames' % (vid, frame_index)), 
                frame_index, fps, time_len)
            
            # vis first frame
            cmd = 'cp %s %s' % (
                osp.join(data_dir, 'JPEGImages', '%s_frame%06d.jpg' % (vid, frame_index)),
                osp.join(clip_dir, '%s_frame%06d/key_frame.jpg' % (vid, frame_index)))
            os.system(cmd)

            clip_list = sorted(glob.glob(
                osp.join(clip_dir, '%s_frame%06d/frames' % (vid, frame_index), '*.jpg')))
            print(len(clip_list))
            write_mp4(clip_list, dst_file, 30)

            # # vis the orig frame rate
            # cmd = 'cp %s %s' % (
            #     osp.join(rgb_dir, vid, 'frame%06d.jpg' % (start_frame_orig)),
            #     osp.join(clip_dir, '%s_frame%06d_copy.jpg' % (vid, frame_index))
            # )
            # print(cmd)
            # os.system(cmd)


def extract_all_vidoes():
    meta_file = os.path.join(data_dir, 'VideoMeta', 'meta_100DOH_video.txt')
    df = pandas.read_csv(meta_file, delim_whitespace=True, index_col=False)

    video_list = glob.glob(osp.join(vid_dir, '*.mp4'))
    vid_id_list = [osp.basename(e).split('.')[0] for e in video_list]
    
    for vid in vid_id_list:
        src_file = osp.join(vid_dir, vid)
        dst_file = osp.join(rgb_dir, vid)
        vid_index = vid.split('_v_')[-1]
        fps = df[df['video_id'] == vid_index].head(1)['frame_rate'].values[0]
        print(fps)
        os.system('rm -r %s' % dst_file)
        os.makedirs(dst_file, exist_ok=True)
        cmd = "ffmpeg -i {:s}.mp4 -vf fps={:s} -qscale:v 2 {:s}/frame%06d.jpg".format(
            src_file, fps, dst_file)
        cmd += ' -hide_banner -loglevel error'
        print(cmd)
        os.system(cmd)
        check(vid)
        

def check(vid):
    frame_list = glob.glob(osp.join(dst_dir, vid, '*.jpg'))
    label = vid.split('_')[0]
    shot_list = [line.strip() for line in open(osp.join(data_dir, 'shot', label, vid + '_shot_info.txt'))]
    if len(frame_list) != len(shot_list):
        print(vid, len(frame_list), len(shot_list))


# # ffmpeg -i test.mp4 -vf fps=0.5 -qscale:v 2 test/frame%06d.jpg 



def ffmpeg_video(fname, src_dir):
    src_list_dir = osp.join(src_dir, '%02d.jpg')
    if not osp.exists(osp.dirname(fname)):
        os.makedirs(osp.dirname(fname), exist_ok=True)
    # if osp.exists(fname + '.mp4'):
    #     os.system('rm %s.mp4' % (fname))
    cmd = 'ffmpeg -framerate 30 -i %s -c:v libx264 -pix_fmt yuv420p %s.mp4' % (src_list_dir, fname)
    # if quiet:
    cmd += ' -hide_banner -loglevel error'

    # cmd = 'ffmpeg -f image2 -r 30 -i %s -vcodec libx264 -crf 18  -pix_fmt yuv420p %s.mp4' % (src_list_dir, fname)
    # cmd = 'ffmpeg -r 30 -f image2 -i %s -vcodec libx264 -crf 25  -pix_fmt yuv420p %s.mp4' % (src_list_dir, fname)
    print(cmd)
    os.system(cmd)


def preview():
    video_list = glob.glob(osp.join(clip_dir, '*'))
    vid_id_list = [osp.basename(e).split('.')[0] for e in video_list]
    valid = 0
    for vid in vid_id_list:
        if has_contact(vid):
            valid += 1
            print(vid)
            src_file = osp.join(clip_dir, vid, 'clip.mp4')
            dst_file = osp.join(clip_dir.replace('100doh_clips', '100doh_preview'), vid)
            cmd = 'cp %s %s.mp4' % (src_file, dst_file)
            os.system(cmd)
    print('total: ', valid, valid / len(vid_id_list))


def has_contact(index):
    if not osp.exists(os.path.join(data_dir, 'Annotations', index + '.xml')):
        return False
    with open(os.path.join(data_dir, 'Annotations', index + '.xml'), 'r') as fid:
        root = ET.parse(fid).getroot()
    for obj in root.findall('object'):
        if obj.find('name').text != 'hand':
            continue
        if obj.find('contactstate').text == '3':
            return True
    return False


if __name__ == '__main__':
    # download_videos()
    # extract_all_vidoes()
    # extract_shot()
    extract_key_frames()
    preview()