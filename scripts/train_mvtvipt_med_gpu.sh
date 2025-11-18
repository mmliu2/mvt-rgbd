#!/bin/bash

cd MVT
python tracking/train_depth.py \
    --script mobilevitvitp_track_depth \
    --config mvtvipt_MED_256_128x1_depthtrack \
    --save_dir ../output \
    --mode single