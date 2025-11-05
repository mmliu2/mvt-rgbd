moved output/, data/, pretrained_model/ to outside of MVT/

mvt-rgbd/
    data/
        * download depthtrack data with scripts/download_depthtrack.sh
    pretrained_models/
        * download mobilevit_s.pt with scripts/download_mobilevit_s.sh
    MVT/
        * setup paths with scripts/setup.sh
        * run experiments with scripts/train*.sh
        * todo: plotting scripts
    scripts/
