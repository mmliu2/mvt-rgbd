#!/bin/bash

cd MVT
python tracking/train_depth.py \
    --script mobilevit_track_det \
    --config mvtdet_256_128x1_depthtrack \
    --save_dir ../output \
    --mode single