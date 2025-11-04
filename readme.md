# or training segmentations

python3 train.py --batch_size 16 --epochs 10 --device cuda --save-name snap-2


# for training heads

python3 train-box.py \
  --annotations annotation.json \
  --images-root dasatet/COD10K-v2/Train/Images/camo_images \
  --batch-size 2 \
  --epochs 200 \
  --lr 1e-5 \
  --device cuda \
  --save-dir models/box_test \
  --export-dir "models/heads" \
  --num-classes 69


curl -L -o dasatet/archive.zip   https://www.kaggle.com/api/v1/datasets/download/ismailelomarialaoui/cod10k



unzip dasatet/archive.zip -d dasatet