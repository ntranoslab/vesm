gpu=$1
model=$2
ckt=$3
data=$4

echo Running inference for $model of $ckt on $data 

CUDA_VISIBLE_DEVICES=$gpu python inference.py -m $model --ckt $ckt  -d $data 


echo Done 

