#!/bin/bash

cd MVT
python tracking/train_depth.py \
    --script mobilevit_track_vipt \
    --config mvtvipt_MED_256_128x1_depthtrack \
    --save_dir ../output \
    --mode single