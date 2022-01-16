# Getting Started

This page provides tutorials about the basic usage of mmderain. 

## Prerequisites

- Linux (it seems that Windows officially supported by mmcv, not so sure)
- Python >= 3.6
- PyTorch >= 1.5

Install some requirements before use

```bash
pip install -r requirements
```

## Training

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

```bash
# single-gpu testing
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]

# multi-gpu testing
./dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```