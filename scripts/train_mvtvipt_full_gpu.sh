#!/bin/bash

cd MVT
python tracking/train_depth.py \
    --script mobilevit_track_depth \
    --config mvtvipt_256_128x1_depthtrack \
    --save_dir ../output \
    --mode single