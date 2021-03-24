# Training_Perusat
Dataset Perusat


How to run
----------
The dataset is organized in the folloing way::
::

        ├── data_HR
        │   ├── test
        │           ├── images
        │           └── masks
        │   └── val
        │           ├── images
        │           └── masks
        │   └── train
        │           ├── images
        │           └── masks
        ├── data_LR
        │   ├── test
        │           ├── images
        │           └── masks
        │   └── val
        │           ├── images
        │           └── masks
        │   └── train
        │           ├── images
        │           └── masks
        ├── logs_LR
        │   ├── mapping
        │           ├── 
        ├── predictions
        ├── history
        │ ......................

# Segmentation_water_bodies_Peru
Dataset Perusat--- HR
Dataset Sentinel_l--- LR



Run_HR: 
1. python train_HR.py
2. python plotting.py  (need path roots)

Run_LR: 
1. python train_LR.py
2. python plotting.py  (need path roots)

Model Combined Parallel: 
1. python train_paral.py
2. python plotting.py  (need path roots)


