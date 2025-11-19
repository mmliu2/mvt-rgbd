#!/bin/bash

cd MVT
python tracking/test_depth.py \
    --dataset_name depthtrackmini_rgb \
    --tracker_name mobilevit_track_vipt \
    --tracker_param mobilevit_256_128x1_got10k_ep100_cosine_annealing \
    --dte=False