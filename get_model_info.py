import argparse

from mmcv import Config
from torchinfo import summary

from mmderain.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a editor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='input image size')
    parser.add_argument(
        '--col-names',
        type=str,
        nargs='+',
        default=['input_size', 'output_size', 'num_params'],
        help='columns to show in the output')
    parser.add_argument(
        '--depth',
        type=int,
        default=5,
        help='depth of nested layers to display'
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif 1 < len(args.shape) <= 4:
        input_shape = ((1, 3) + tuple(args.shape))[-4:]
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(f'{model.__class__.__name__} is currently not supported ')

    summary(model, input_shape, depth=args.depth, col_names=args.col_names)


if __name__ == '__main__':
    main()
