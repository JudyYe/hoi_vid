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


? multiple object segmentation??? 
```
obj/
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

3. Run STCN
construct backward folder from forward pass
python run_doh.py


4. preview 