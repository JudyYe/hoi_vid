#!/bin/bash
 
set -x 

# export ODIR='/home/yufeiy2/scratch/result/3rd_cache/by_obj'
# export PDIR=/home/yufeiy2/scratch/result/3rd_cache/by_ppl
# export FDIR=/home/yufeiy2/scratch/result/3rd_cache/by_seq
# export DATA_DIR='/home/yufeiy2/scratch/result/3rd_cache'
# export RAWDIR='/home/yufeiy2/scratch/data/3rd'

export ODIR='/home/yufeiy2/scratch/result/1st_cache/by_obj'
export PDIR=/home/yufeiy2/scratch/result/1st_cache/by_ppl
export FDIR=/home/yufeiy2/scratch/result/1st_cache/by_seq
export DATA_DIR='/home/yufeiy2/scratch/result/1st_cache'
export RAWDIR='/home/yufeiy2/scratch/data/1st'


seqname=$1

# # extract first frame
python -m preprocess.hoi_det_custom $seqname
# # put to STCN format
python -m preprocess.cvt_to_stcn $seqname*

# # # track by STCN
# cd ../STCN
# python run_doh.py --folder ../../output/100doh_detectron/by_obj --seq $seqname*
# python run_doh.py --folder ../../output/100doh_detectron/by_ppl --seq $seqname*

cd ../STCN
python run_doh.py --folder ${ODIR} --seq $seqname*
python run_doh.py --folder ${PDIR} --seq $seqname*


# # # evaluate and visualize
cd - 
python -m preprocess.eval_vis $seqname*


# reconstruct hand
# * is taken cared
bash preprocess/hand_det_w_gt.sh $seqname

# # ihoi
# # python scripts/generic_batch.py  bash scripts/run_ihoi.sh