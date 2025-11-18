#!/bin/bash

cd MVT
python tracking/test_depth.py \
    --dataset_name depthtrackmini \
    --tracker_name mobilevitvitp_track_depth \
    --tracker_param mvtvitp_256_128x1_depthtrack \
    --dte=True