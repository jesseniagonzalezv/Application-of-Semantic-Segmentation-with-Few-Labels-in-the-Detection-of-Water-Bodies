# Segmentation_water_bodies_Peru
----------
### Details of the project.pdf

This project consist of two dataset::
- Sentinel: data_HR
- PeruSat_1: data_VHR

Pd: The images must be entered in the following format: CHxHxW.

How to run
----------
The dataset is organized in the folloing way::
::

        ├── data_HR
        │   ├── test_HR
        │           ├── images
        │           └── masks
        │   └── train_val_HR
        │           ├── images
        │           └── masks
        ├── data_VHR
        │   ├── test_850
        │           ├── images
        │           └── masks
        │   └── train_val_850
        │           ├── images
        │           └── masks
        ├── logs_HR
        │   ├── mapping
        │           ├── 
        ├── predictions_HR
        ├── history_HR
        ├── logs_VHR
        │   ├── mapping
        │           ├── 
        ├── predictions_VHR
        ├── history_VHR
        ├── logs_paral
        │   ├── mapping
        │           ├── 
        ├── predictions_paral
        ├── history_paral
        ├── logs_seq
        │   ├── mapping
        │           ├── 
        ├── predictions_seq
        ├── history_seq
        │ ......................


### Run each Model:
        1. In train_model.sh there is a example how the models were trained.

        Run HR Sentinel:
        #python train.py --batch-size 8 --lr 1e-3  --n-epochs 40  --model 'UNet11' --dataset-path 'data_HR' --dataset-file 'HR' --train-val-file 'train_val_HR' --test-file 'test_HR'
        #python plotting.py --out-file 'HR' --stage 'test' --name-file '_100_percent_HR' --name-model 'UNet11' --epochs 40 --count 613

        Run VHR Perusat: 
        python train.py --percent 0.08 --batch-size 4 --lr 1e-3  --n-epochs 40  --model 'UNet11' --dataset-path 'data_VHR' --dataset-file 'VHR' --train-val-file 'train_val_850' --test-file 'test_850'
        python plotting.py --out-file 'VHR' --stage 'test' --name-file '_8_percent_VHR' --name-model 'UNet11' --epochs 40 --count 94
        
        Run Parallel
        python train_paral.py --percent 0.08 --batch-size 4 --n-epochs 40 --n-steps 34  --lr 1e-3   --modelVHR 'UNet11' --dataset-path-HR 'data_HR'  --dataset-path-VHR 'data_VHR' --train-val-file-HR 'train_val_HR'   --test-file-HR 'test_HR' --train-val-file-VHR 'train_val_850' --test-file-VHR 'test_850'
        python plotting.py --out-file 'paral' --stage 'test' --name-file '_8_percent_paral' --name-model 'UNet11' --epochs 40 --count 94

        Run sequential
        python train_seq.py --percent 0.08 --batch-size 4 --n-epochs 40 --n-steps 34  --lr 1e-3   --modelVHR 'UNet11' --dataset-path-HR 'data_HR'  --dataset-path-VHR 'data_VHR' --train-val-file-HR 'train_val_HR'   --test-file-HR 'test_HR' --train-val-file-VHR 'train_val_850' --test-file-VHR 'test_850'
        python plotting.py --out-file 'seq' --stage 'test' --name-file '_8_percent_seq' --name-model 'UNet11' --epochs 40 --count 94

### To run  a cross validation use:
        1. bash test_all.sh  
