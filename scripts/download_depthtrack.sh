#!/bin/bash

# downloads train val splits used for DeT, and full test set

python scripts/download_depthtrack.py data
# python scripts/download_depthtrack.py data --mini