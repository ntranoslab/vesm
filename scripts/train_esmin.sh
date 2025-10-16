model=$1
gpu=$2
batch_size=12

echo exp $model on device $gpu
CUDA_VISIBLE_DEVICES=$gpu python train.py -m $model -c esmin -b $batch_size