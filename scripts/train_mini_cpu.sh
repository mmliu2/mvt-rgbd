cd MVT
python tracking/train_depth.py \
    --script mobilevit_track_depth \
    --config mvt_rgbd_MINI_256_128x1_depthtrack \
    --save_dir ./../output \
    --mode single