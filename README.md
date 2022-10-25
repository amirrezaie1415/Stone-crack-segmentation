# Crack Segmentation in Stone

This repo includes the codes for segmentation of crack pixels in scanned stone samples. 

# How to use it?

## 1. Clone repository

All necessary data and codes are inside the ``src`` directory. 

## 2. Install Conda or Miniconda

Link to install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html

Link to install miniconda: https://docs.conda.io/en/latest/miniconda.html

## 3. Create a conda environment 


Run the following commands in your terminal to install and activate the environment.

```bash
conda env create -f environment.yml
```

```bash
conda activate stone_crack_detection
```

## 4. Create the dataset directory


Stone-crack-segmentation
└───dataset
    └───train
    └───train_GT
    └───test
    └───test_GT
    └───valid
    └───valid_GT
└───models
└───logs
└───src


## 5. Train a model

To train a deep model you can run the following command:
```bash
    python run.py --model_type=TernausNet16 --lr=1e-4 --weight_decay=0 --num_epochs=50 --pretrained=1  --batch_size=32
```

## 6. Inference

TO DO


# Citation

The src codes in this repo are modified versions of the codes in the [Deep DIC Crack](https://github.com/amirrezaie1415/Deep-DIC-Crack) repo. 

If you find this implementation useful, please cite us as:
```
@article{REZAIE2020120474,
title = {Comparison of crack segmentation using digital image correlation measurements and deep learning},
journal = {Construction and Building Materials},
volume = {261},
pages = {120474},
year = {2020},
issn = {0950-0618},
doi = {https://doi.org/10.1016/j.conbuildmat.2020.120474},
url = {https://www.sciencedirect.com/science/article/pii/S095006182032479X},
author = {Amir Rezaie and Radhakrishna Achanta and Michele Godio and Katrin Beyer},
keywords = {Crack segmentation, Digital image correlation, Deep learning, Threshold method, Masonry},
}
```
