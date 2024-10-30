[![Documentation Status](https://readthedocs.org/projects/segmentmytiff/badge/?version=latest)](https://segmentmytiff.readthedocs.io/en/latest/?badge=latest) [![build](https://github.com/DroneML/segmentmytiff/actions/workflows/build.yml/badge.svg)](https://github.com/DroneML/segmentmytiff/actions/workflows/build.yml) [![cffconvert](https://github.com/DroneML/segmentmytiff/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/DroneML/segmentmytiff/actions/workflows/cffconvert.yml)

## How to use segmentmytiff

Segment (georeferenced) raster data in an interactive fashion. Retrain models in seconds. Only small amounts of labeled data necessary because of our use of pretrained base models as feature extractors.

The project setup is documented in [project_setup.md](devdocs/project_setup.md).

## Installation

To install segmentmytiff from GitHub repository, do:

```console
git clone git@github.com:DroneML/segmentmytiff.git
cd segmentmytiff
python -m pip install .
```

## Train a feature extraction model

To train a feature extraction model run the script "train_model.py" in this repo:
```bash
python ./src/segmentmytiff/utils/train_model.py -r ../monochrome_flair_1_toy_dataset_flat/ --train_set_limit 10
```
This assumes a 'flat', grayscale, version of the FLAIR1 dataset is present at the selected root location.
```
root
- train
    - input
        - IMG_061946_0.tif
        - IMG_061946_1.tif
        - ...
    - labels
        - MSK_061946_0.tif
        - ...    
```
Use the script 'monochromize.py' to create greyscale (single band) tifs for every multiband tif in a source folder:
```bash
python ./src/segmentmytiff/utils/monochromize.py -i ../flair_1_toy_dataset/ -o ../monochrome_flair_1_toy_dataset/
```

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
