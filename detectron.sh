#/bin/bash

git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
pip install Pillow==6.2.2 pyyaml==5.1
pip install albumentations
pip install -e detectron2_repo