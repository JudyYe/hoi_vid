set -x 

# conda activate stcn

cd ../STCN

python run_doh.py --folder ../output/100doh_detectron/by_obj
python run_doh.py --folder ../output/100doh_detectron/by_ppl