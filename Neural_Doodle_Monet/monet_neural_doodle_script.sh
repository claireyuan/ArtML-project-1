#!/bin/bash
python neural_doodle.py --nlabels 3 --style-image monet_art1.png --style-mask monet_art1_mask.png --target-mask monet_style_transfer_mask.png --content-image monet_style_transfer.png --target-image-prefix generated_monet
