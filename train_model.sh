#!/bin/ bash
# 

echo run models

#HR Sentinel
#python train.py --batch-size 8 --lr 1e-3  --n-epochs 40  --model 'UNet11' --dataset-path 'data_HR' --dataset-file 'HR' --train-val-file 'train_val_HR' --test-file 'test_HR'
#python plotting.py --out-file 'HR' --stage 'test' --name-file '_100_percent_HR' --name-model 'UNet11' --epochs 40 --count 613


# VHR PeruSat-1
python train.py --percent 0.08 --batch-size 4 --lr 1e-3  --n-epochs 40  --model 'UNet11' --dataset-path 'data_VHR' --dataset-file 'VHR' --train-val-file 'train_val_850' --test-file 'test_850'
python plotting.py --out-file 'VHR' --stage 'test' --name-file '_8_percent_VHR' --name-model 'UNet11' --epochs 40 --count 94


#### Parallel
python train_paral.py --percent 0.08 --batch-size 4 --n-epochs 40 --n-steps 34  --lr 1e-3   --modelVHR 'UNet11' --dataset-path-HR 'data_HR'  --dataset-path-VHR 'data_VHR' --train-val-file-HR 'train_val_HR'   --test-file-HR 'test_HR' --train-val-file-VHR 'train_val_850' --test-file-VHR 'test_850'
python plotting.py --out-file 'paral' --stage 'test' --name-file '_8_percent_paral' --name-model 'UNet11' --epochs 40 --count 94



#### Seq
python train_seq.py --percent 0.08 --batch-size 4 --n-epochs 40 --n-steps 34  --lr 1e-3   --modelVHR 'UNet11' --dataset-path-HR 'data_HR'  --dataset-path-VHR 'data_VHR' --train-val-file-HR 'train_val_HR'   --test-file-HR 'test_HR' --train-val-file-VHR 'train_val_850' --test-file-VHR 'test_850'
python plotting.py --out-file 'seq' --stage 'test' --name-file '_8_percent_seq' --name-model 'UNet11' --epochs 40 --count 94



