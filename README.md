Satellite Image Difference
==============================
## Description

With a given two images of the same place from the different dates we need to figure out binary mask with changes

[Selim's pretrained models](https://github.com/selimsef/xview2_solution/releases/tag/0.0.1)

## Train
1. Put data, pretrained models in a proper folders
2. Run the training proccess
```
python train.py config/config_unet++_resnext50.json
python train.py config/config_siamese_seresnext50.json
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md
    ├── data
    │   ├── Images                 <- Original 4 channel images from first and second dates.
    │   ├── Images_composit        <- Composed 8 channel images from original images.
    │   ├── mask                   <- Binary masks of images differences.
    │   ├── Rucode.xls             <- Table to match images from the same location.
    │   └── sample_submission.csv  <- Sample submission file.
    │
    ├── models             <- Trained and serialized models
    │   ├── pretrained     <- Pretrained models for siamese models from Selim.
    │   └── saved          <- Trained on the competition data models.
    |
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── logs               <- Logs of training process: loss, IoU
    |
    ├── config             <- Configuration files for each type of model
    |
    ├── predicted_masks    <- Predicted probability masks in pickle format 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    │                         `pip install -r requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── train.py           <- Training process with evaluation on the end.
    ├── eval.py            <- Mask evaluation using the trained model.
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── models.py      <- Siamese models from Selim
    │   ├── dataset.py     <- Dataset class for satellite images
    │   └── utils.py       <- Small preprocess functions like normalization and decoding
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
