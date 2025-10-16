gpu=$1
data=$2
round_name=$3
model=$4
partition=0

echo Running inference for $model of $round_name on $data 

cd data

CUDA_VISIBLE_DEVICES=$gpu python infer_llrs.py -r $round_name -m $model -d $data -p $partition

cd ..

echo Done 

