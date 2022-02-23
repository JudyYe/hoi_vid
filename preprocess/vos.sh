#!/bin/bash
 
set -x 

seqname=$1

# extract first frame
python hoi_det.py $seqname
# put to STCN format
python cvt_to_stcn.py $seqname*

# track by STCN
cd ../../STCN
python run_doh.py --folder ../../output/100doh_detectron/by_obj --seq $seqname*
python run_doh.py --folder ../../output/100doh_detectron/by_ppl --seq $seqname*


# evaluate and visualize
cd ../src/preprocess
python eval_vis.py $seqname*


# reconstruct hand
# * is taken cared
bash preprocess/hand_det_w_gt.sh $seqname

# ihoi
# python scripts/generic_batch.py  bash scripts/run_ihoi.sh