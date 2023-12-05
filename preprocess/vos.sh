#!/bin/bash
 
set -x 

export DET_DIR='/private/home/yufeiy2/Packages/detectron2'   # your detectron2 path
# export DATA_DIR='/private/home/yufeiy2/scratch/result/1st_cache'  # output path
# export RAWDIR='/private/home/yufeiy2/scratch/result/RAW_WILD/'    # input sequence path

export RAWDIR='/private/home/yufeiy2/scratch/result/wild/raw'    # input sequence path
export DATA_DIR='/private/home/yufeiy2/scratch/result/wild/raw_cache'
# export ODIR='/private/home/yufeiy2/scratch/result/3rd_cache/by_obj'
# export PDIR=/private/home/yufeiy2/scratch/result/3rd_cache/by_ppl
# export FDIR=/private/home/yufeiy2/scratch/result/3rd_cache/by_seq
# export DATA_DIR='/private/home/yufeiy2/scratch/result/3rd_cache'
# export RAWDIR='/private/home/yufeiy2/scratch/data/3rd'

# intermediate parameters
export ODIR=${DATA_DIR}/by_obj
export PDIR=${DATA_DIR}/by_ppl
export FDIR=${DATA_DIR}/by_seq


seqname=$1

# extract first frame
# python -m preprocess.hoi_det_custom $seqname
# python -m preprocess.hoi_sam_custom $seqname
# put to STCN format
python -m preprocess.cvt_to_stcn $seqname*

# track by STCN

cd ../STCN
python run_doh.py --folder ${ODIR} --seq $seqname*
python run_doh.py --folder ${PDIR} --seq $seqname*


# evaluate and visualize
cd - 
python -m preprocess.eval_vis $seqname*


# reconstruct hand
# * is taken cared
bash preprocess/hand_det_w_gt.sh $seqname

# # ihoi
# # python scripts/generic_batch.py  bash scripts/run_ihoi.sh