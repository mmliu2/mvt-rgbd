#!/bin/bash

# evalulate pretrained mvt on only rgb images

cd MVT
python tracking/test.py \
    --dataset_name depthtrack_rgb \
    --tracker_name mobilevit_track \
    --tracker_param mobilevit_256_128x1_got10k_ep100_cosine_annealing