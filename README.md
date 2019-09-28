

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
        ├── logs
        │   ├── mapping
        │           ├── final_layer
        ├── predictions
        │ ......................

# Segmentation_water_bodies_Peru
Dataset Perusat


Run: 
1. python train.py
2. python prediction_mask.py

Test the model with the test_dataset:
        testing_model.ipynb
