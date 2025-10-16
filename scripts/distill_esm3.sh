gpu=$1
model=esm3

echo Distilling to $model on device $gpu
CUDA_VISIBLE_DEVICES=$gpu python train.py -m $model -c distillation_esm3 -b 4 -v 2
    