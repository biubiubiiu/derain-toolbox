# Dataset Preparation

Derain-toolbox provides builtin support for three dataset formats, those are:

* [DerainPairedDataset](../mmderain/datasets/derain_paired_dataset.py)
* [DerainUnpairedDataset](../mmderain/datasets/derain_unpaired_dataset.py)
* [DerainFilenameMatchingDataset](../mmderain/datasets/derain_filename_matching_dataset.py)

It is recommended to symlink the dataset root to `../data`. If your folder structure is different, you may need to change the corresponding paths in config files.

## Load data with `DerainPairedDataset`

`DerainPairedDataset` expects the dataset in the following structure:

    ${data_root}
    ├── train
    ├── val
    ├── test

`DerainPairedDataset` is used as the default dataset format. It assumes that rainy image and the corresponding label are concatenated in the width dimension, i.e. :

    ┌───────────────────┬──────────────────┐
    │                   │                  │
    │                   │                  │
    │                   │                  │
    │         A         │         B        │
    │                   │                  │
    │                   │                  │
    │                   │                  │
    └───────────────────┴──────────────────┘

The relative order of rainy and background image should be specified in the config files:

```python
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        key='lq,gt',  # relative order of rainy image (marked by "lq") and label ("gt"), seperated by comma
    ),
    ...
]
```

## Load Data with `DerainUnpairedDataset`

`DerainUnpairedDataset` expects the dataset in the following structure:

    ${data_root}
    ├── train
    │   ├──${dataroot_a}
    │   ├──${dataroot_b}
    ├── val
    │   ├──${dataroot_a}
    │   ├──${dataroot_b}
    ├── test
    │   ├──${dataroot_a}
    │   ├──${dataroot_b}

`DerainUnpairedDataset` is used for unpaired training, where the rainy images and the background images are unpaired.

## Load data with `DerainFilenameMatchingDataset`

`DerainFilenameMatchingDataset` expects the dataset in the following structure:

    ${data_root}
    ├── train
    │   ├──${lq_folder}
    │   ├──${gt_folder}
    ├── val
    │   ├──${lq_folder}
    │   ├──${gt_folder}
    ├── test
    │   ├──${lq_folder}
    │   ├──${gt_folder}

`DerainPairedDataset` is used when rainy images and corresponding labels are in different folders. Rainy images and labels are paired by filename matching, including full matching and prefix matching.

- Example for filename full matching:

        ${data_root}
        ├── train
        │   ├──${lq_folder}
        │   │   ├──1.png
        │   │   ├──2.png
        │   ├──${gt_folder}
        │   │   ├──1.png
        │   │   ├──2.png
        ├── test
        │   ├──${lq_folder}
        │   │   ├──1.png
        │   │   ├──2.png
        │   ├──${gt_folder}
        │   │   ├──1.png
        │   │   ├──2.png

- Example for filename prefix matching

        ${data_root}
        ├── train
        │   ├──${lq_folder}
        │   │   ├──1_1.png
        │   │   ├──1_2.png
        │   │   ├──1_3.png
        │   ├──${gt_folder}
        │   │   ├──1.png
        ├── test
        │   ├──${lq_folder}
        │   │   ├──1_1.png
        │   │   ├──1_2.png
        │   │   ├──1_3.png
        │   ├──${gt_folder}
        │   │   ├──1.png
