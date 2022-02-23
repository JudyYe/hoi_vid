set -x 


seq=$1

cd ~/hoi/

python -m ihoi.scripts.demo --seq ${seq} \
    -e output/aug/pifu_MODEL.DECPixCoord/ \
    --out /checkpoint/yufeiy2/vhoi_out/ihoi_out_obman \
    --sdf_out /checkpoint/yufeiy2/vhoi_out/database/DAVIS/Sdf_obman/Full-Resolution/ \


python -m ihoi.scripts.demo --seq ${seq}

out_dir=/checkpoint/yufeiy2/vhoi_out/database/DAVIS/Sdf/Full-Resolution
odir=/checkpoint/yufeiy2/vhoi_out/100doh_detectron/by_obj/Ihoi

mkdir -p $odir/${seq}
cp -r $out_dir/r${seq}/* $odir/$seq/


