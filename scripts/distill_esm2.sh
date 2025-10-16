model=$1
gpu=$2
batch_size=16

echo Distilling to $model on device $gpu
CUDA_VISIBLE_DEVICES=$gpu python train.py -m $model -c distillation_esm2 -b $batch_size
    