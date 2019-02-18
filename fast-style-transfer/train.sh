"""
Training script for fast style transfer with hyperparameters. 

First, we run this script with the photo as the test (content) image and the 
self-portrait as the style image. Then, we run this script again with the 
intermediate result as the content image and another example of the artist's 
work as the style.
"""

set -x

python style.py \
  --checkpoint-dir model \
  --style picassoStyle.jpg \
  --test picassoResult.jpg \
  --test-dir results/first \
  --epochs 2 \
  --batch-size 4 \
  --checkpoint-iterations 2000 \
  --content-weight 10 \
  --style-weight 100 \
  --learning-rate 0.001 \
  2>&1 | tee -a picassoNew.log
  
