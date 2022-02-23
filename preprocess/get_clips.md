1. use extract_100doh.py:download_videos() extract_key_frames to download and get some clips
```python extract_100doh.py```
the clip will be saved to 
```
output/100doh_clips/
    diy_xcvdtw_frame234325/
        clip.mp4
        key_frame.jpg
        frames/
            01.jpg - xx.jpg
```




2. Given GT bbox, get object and hand segmentation for the first frame of all the videos with `hoi_det.py`
The JPEGImages and Annotation will be saved in the form ready for STCN to process (only forward pass). 

list all clips and feed them through vos.sh one by one
```
python batch_vos.py
```
`vos.sh`: 
- extract first frame
- put to STCN format
- track by STCN
- evaluate and visualize



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
```
