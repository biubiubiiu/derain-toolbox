# Getting Started

This document provides a brief intro of the usage of builtin command-line tools in derain-toolbox.

For a tutorial that involves actual coding with the API, see our Colab Notebook which covers how to run inference with an existing model, and how to train a builtin model on a custom dataset.

## Training

Two scripts in "train.py" and "dist_train.sh" are made to train all the configs provided in this repo. You may want to use it as a reference to write your own training script.

To train a model, first setup the corresponding datasets following [dataset_preparation](dataset_preparation.md), then run:

```bash
# single-gpu training
python train.py ${CONFIG_FILE} [optional arguments]

# multi-gpu training
./dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--no-validate`: By default, the codebase will perform evaluation every k iterations during the training. To disable this behavior, use `--no-validate`

## Testing

The evaluate a model's performance, use:

```bash
# single-gpu testing
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]

# multi-gpu testing
./dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```