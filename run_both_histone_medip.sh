#!/bin/bash

for i in {0..49}
do
    echo $i
    cd /data1/test_attention/
    #cp /home/penglab/fly/Deep_learning/iDeepV/All_data/motif_discovery/$i/* /home/penglab/fly/Deep_learning/DeepS/DBP/datasets/
    cp /data1/fly/CpG_data/$i/*ll dataset_differ_mer/
    echo $i "copy done!!! Starting processing..."
    #THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python Merge_Onehot_Dict_add_modify_tmp.py --train=True --data_file=dataset_tmp/Train_all --model_dir=model --dataset_num $i
    THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python MHCpG.py --train=True --data_file=dataset_differ_mer/Train_all --model_dir=model --dataset_num $i
    echo $i "done"


done
