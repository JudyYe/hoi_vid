## What is it? 

- Input: Hand-object interaction (HOI) videos with hand-object bounding box in the first frame. 

- Output: per-frame hand-object masks, hand poses.


## Installation 
We rely on [STCN](https://github.com/hkchengrex/STCN#), a great video object segmentation system: 
```
cd ..
git clone https://github.com/hkchengrex/STCN.git
cd - 
cp scripts/run_doh.py ../STCN/
```



### Data Formats
Prepare your own sequence into the following format:
```
$seq1/
    iamges/%04d.png
    bbox.json # {'hand': [x1, y1, x2, y2], 'obj': [x1, y1, x2, y2]}
$seq2/
    iamges/%04d.png
    bbox.json 
```


<details><summary>For 100DOH dataset</summary>

The script downloads some videos from 100DOH and extracts short clips around its key frames.  Download from 100DOH dataset: use `extract_100doh.py:download_videos()`, `extract_key_frames()` to download and get some clips
`python extract_100doh.py`.
the clip will be saved to 
```
output/100doh_clips/
    diy_xcvdtw_frame234325/
        # user provided bbox.json: 
        {'obj':  [x1, y1, x2, y2]], 'hand': [x1, y1, x2, y2]}
        clip.mp4
        key_frame.jpg
        frames/
            01.jpg - xx.jpg
```

</details>


## Process Sequence
Given the bounding box of hand and object in the first frame from GT annotation, `vos.sh` tracks both hand and object, then gets their masks, reconstructs hand, and finds correspondence between multiple hand and objects.


2. One-click preprocess: After change `DET_DIR, DATA_DIR, RAWDIR` in `vos.sh`
<!-- The JPEGImages and Annotation will be saved in the form ready for STCN to process (only forward pass).  -->

```
sh vos.sh $seq
```
- extract masks in the first frame  `hoi_det.py`
- put to STCN format
- track by STCN
- evaluate tracking quality (by mask IoU between first frame and tracking forward and backward) and visualize masks.
- find coresponding hand and reconstruct hand by frank mocap

Batchify script: list all clips and feed them through vos.sh one by one
```
python batch_vos.py
```

<!-- 
--- 
`100doh_detectron`
? multiple object segmentation??? 
```
100doh_detectron/by_obj/
    JPEGImages/
        video1_o1/
            00000.png....
    Annotations/
        videos1_o1/
            00000.png
            00000.mat
```

then multiple person segmetation from the obove:
```
ppl/
    JPEGImages/
        video1_p1/
            00000.png....
    Annotations/
        videos1_p1/
            00000.png
``` -->
