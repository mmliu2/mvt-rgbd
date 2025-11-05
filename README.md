moved output/, data/, pretrained_model/ to outside of MVT/

mvt-rgbd/
    data/
        * download depthtrack data with scripts/download_depthtrack.sh
    output/
        * download pretrained mvt with scripts/download_mvt_checkpoints.sh
    MVT/
        * download mobilevit_s and setup paths with scripts/setup_mvt.sh
        * run experiments with scripts/train*.sh
        * todo: plotting scripts
    scripts/
