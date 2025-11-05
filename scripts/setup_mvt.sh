wget -qnc https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt -O MVT/pretrained_models/mobilevit_s.pt

python3 MVT/tracking/create_default_local_file.py \
    --workspace_dir . \
    --data_dir ./../data \
    --save_dir ./../output
