#!/bin/bash

# evalulate mvt-det on rgbd images

cd MVT
python tracking/test.py \
    --dataset_name depthtrack \
    --tracker_name mobilevit_track_det \
    --tracker_param mvtdet_256_128x1_depthtrack