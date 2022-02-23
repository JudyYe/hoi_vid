import ast, operator
import numpy as np
import os
import os.path as osp
import imageio
import datetime

def write_mp4(video, save_file, fps=10, clear=True):
    tmp_dir = save_file + '.tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    for t, image in enumerate(video):
        if isinstance(image, str):
            # os.system('cp %s %s' % (image, osp.join(tmp_dir, '%03d.jpg' % t)))
            os.system('cp %s %s' % (image, osp.join(tmp_dir, '%03d.%s' % (t, image.split('.')[-1]))))
            ext = '%s' % image.split('.')[-1]
        else:
            imageio.imwrite(osp.join(tmp_dir, '%03d.jpg' % t), image.astype(np.uint8))
            ext = 'jpg'

    if osp.exists(save_file + '.mp4'):
        os.system('rm %s.mp4' % (save_file))
    src_list_dir = osp.join(tmp_dir, '%03d.' + ext )
    cmd = 'ffmpeg -framerate %d -i %s -c:v libx264 -pix_fmt yuv420p %s.mp4' % (fps, src_list_dir, save_file)
    # cmd = '/home/yufeiy2/.local/bin/ffmpeg -framerate %d -i %s -c:v libx264 -pix_fmt yuv420p %s.mp4' % (fps, src_list_dir, save_file)
    cmd += ' -hide_banner -loglevel error'
    print(cmd)
    os.system(cmd)
    if clear:
        cmd = 'rm -r %s' % tmp_dir
        os.system(cmd)


def frame_num_to_time(frame_index, fps):
    """
    frame_index: 1-index
    """
    return (frame_index - 1) / fps + 1/fps/2

def time_to_frame_num(time, fps):
     # (1-1/30) * 30 = 29?? 
    return  time * fps + 1

def cvt_frame_num(src_frame, src_fps, dst_fps):
    """
    @param src_frame: 1-index
    @return: dst_fps: 1-index
    """
    dst_fps = arithmeticEval(dst_fps)
    src_fps = arithmeticEval(src_fps)
    time = frame_num_to_time(src_frame, src_fps) - 1/dst_fps
    dst_frame = time_to_frame_num(time, dst_fps)
    return int(dst_frame)

def extract_frame(src_vid_file, dst_dir, key_frame_num, fps='30', time_len=1):
    """extract the nth key-frame that was decoded in 0.5 fps of an video encoded in 'fps' fps"""
    # dst_dir = os.path.join(clip_dir, '%s_frame%06d/frame' % (vid_name, key_frame_num))
    # dst_file = '%s_frame%06d/frame' % (dst_dir, key_frame_num)
    os.makedirs(dst_dir, exist_ok=True)
    start_time = str(datetime.timedelta(seconds=(key_frame_num - 1) * 2 + 1  - 1/arithmeticEval(fps)))
    time_span = str(datetime.timedelta(seconds=time_len))

    cmd = """ffmpeg -i %s.mp4 -ss %s -t %s -async 1  """ % (src_vid_file, start_time, time_span)
    cmd += """-vf  fps={1} -qscale:v 2  {0}/%02d.jpg""".format(dst_dir, fps)

    # if quiet:
    cmd += ' -hide_banner -loglevel error'
    os.system(cmd)
    return dst_dir



binOps = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod
}

def arithmeticEval (s):
    if isinstance(s, int) or isinstance(s, float):
        return s
    node = ast.parse(s, mode='eval')

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return binOps[type(node.op)](_eval(node.left), _eval(node.right))
        else:
            raise Exception('Unsupported type {}'.format(node))

    return _eval(node.body)