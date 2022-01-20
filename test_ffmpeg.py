import os
import cv2
import numpy as np

from ffmpeg_utils import write_mp4, extract_frame

H = W = 256
fps = 30
T = 4
N = T * fps
canvas = [np.zeros([H, W, 3]).astype(np.uint8) for n in range(N)]

for n in range(N):
    cv2.putText(canvas[n], '%d' % n, (0, 2*H//3), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 0))

write_mp4(canvas, '../output/test', 30)

os.makedirs('../output/test/', exist_ok=True)
cmd = 'ffmpeg -i ../output/test.mp4 -vf fps=0.75 -qscale:v 2 ../output/test/frame%06d.jpg'
os.system(cmd)

extract_frame('../output/test', '../output/test_frame/', 2)
