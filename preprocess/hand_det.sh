set -x

cd ../frankmocap/


# will be saved to ../output/100doh_detectron/by_obj/mocap/
# together with rendered results
seqname=$1
# for plain image, wo hand detector
# xvfb-run -a python -m demo.demo_handmocap \
#    --input_path ../output/100doh_detectron/by_obj/JPEGImages/${seqname}/ \
#    --out_dir ../output/100doh_detectron/by_obj/mocap/${seqname}/ \
#    --save_pred_pkl \

#  xvfb-run -a python -m demo.demo_handmocap \
#      --input_path ../output/100doh_detectron/by_obj/JPEGImages/${seqname}/ \
#      --out_dir ../output/100doh_detectron/by_obj/mocap_ego/${seqname}/ \
#      --save_pred_pkl \
#      --view_type ego_centric \


 xvfb-run -a python -m demo.demo_handmocap \
     --input_path ../output/100doh_detectron/by_seq/hand_box/${seqname}/ \
     --out_dir ../output/100doh_detectron/by_seq/mocap/${seqname}/ \
     --save_pred_pkl \


