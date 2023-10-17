set -x

seqname=$1


# # get hand bbox per frame
# python -m preprocess.find_hand ${seqname}
# # cvt to mocap format
# python preprocess/hand_bbox_to_frankmocap.py ${seqname}

# get hand bbox per frame
python -m preprocess.find_custom_hand ${seqname}
# # cvt to mocap format
# python preprocess/hand_bbox_to_frankmocap.py ${seqname}


# cd /home/yufeiy2/frankmocap/

#  xvfb-run -a python -m demo.demo_handmocap \
#      --input_path ${FDIR}/hand_box/${seqname}/ \
#      --out_dir ${FDIR}/mocap/${seqname}/ \
#      --save_pred_pkl  --renderer_type pytorch3d --no_display --no_video_out \


# # distribute to targeted folder odir
# cd - 
# python preprocess/dist_to_obj.py ${seqname}