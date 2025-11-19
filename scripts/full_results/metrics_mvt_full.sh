#!/bin/bash

cd MVT
python tracking/results_metrics_depthtrack.py \
    --pred_dir='../output/test/tracking_results/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing'