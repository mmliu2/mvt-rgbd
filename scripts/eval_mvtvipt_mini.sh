#!/bin/bash

cd MVT
python tracking/test_depth.py \
    --dataset_name depthtrackmini \
    --tracker_name mobilevit_track_vipt \
    --tracker_param mvtvipt_256_128x1_depthtrack \
    --dte=True