#!/bin/bash

# Ensure destination directory exists
CHECKPOINTS_DIR='output/checkpoints/train/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing'

FILES=(
    "1nwEmq5ZhGYro8TKCX6PrcR-SUQ88c9fS MobileViT_Track_ep0100_state_dict.pt"
    "1OnBj07TbzOn1JeXSffYgq4l4WqUL2WOw MobileViT_Track_ep0100.onnx"
    "1ngoDHI9ip6AUM0EU_rrIXeN2mycgIlru MobileViT_Track_ep0100.pth.tar"
    "1F6iThNlFcyxfDeworOz170BhH2AU4G3h MobileViT_Track_ep0300.mnn"
    "15dI9j7UQc35pcWjD0133eRzLh0P_fRvx MobileViT_Track_ep0300.onnx"
)

# Download each file
for entry in "${FILES[@]}"; do
    set -- $entry
    FILE_ID=$1
    FILENAME=$2

    echo "Downloading ${FILENAME}..."
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt \
        --keep-session-cookies --no-check-certificate \
        "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O- | \
        sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')

    wget --load-cookies /tmp/cookies.txt \
        "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
        -O "${CHECKPOINTS_DIR}/${FILENAME}" && rm -rf /tmp/cookies.txt
done