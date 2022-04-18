set -x

seqname=$1


# get hand bbox per frame
python -m preprocess.find_hand ${seqname}
# cvt to mocap format
python preprocess/hand_bbox_to_frankmocap.py ${seqname}


cd ../frankmocap/

 xvfb-run -a python -m demo.demo_handmocap \
     --input_path ../output/100doh_detectron/by_seq/hand_box/${seqname}/ \
     --out_dir ../output/100doh_detectron/by_seq/mocap/${seqname}/ \
     --save_pred_pkl  --renderer_type pytorch3d --no_display --no_video_out \


# distribute to targeted folder odir
cd ../vhoi
python preprocess/dist_to_obj.py ${seqname}