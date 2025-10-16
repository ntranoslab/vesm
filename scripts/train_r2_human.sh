model=$1
gpu=$2

echo exp $model on device $gpu
CUDA_VISIBLE_DEVICES=$gpu python train.py -m $model -c r2_UPh
    