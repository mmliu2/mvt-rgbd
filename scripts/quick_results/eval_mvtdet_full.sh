#!/bin/bash

cd MVT
python tracking/test.py \
    --dataset_name depthtrackmini \
    --tracker_name mobilevit_track_det \
    --tracker_param mvtdet_256_128x1_depthtrack