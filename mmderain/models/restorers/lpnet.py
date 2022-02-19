import numbers
import os.path as osp

import mmcv
from torch import nn

from mmderain.core import crop_border, psnr, ssim, tensor2img
from mmderain.models.common import gaussian_pyramid

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS


@MODELS.register_module()
class LPNet(BaseModel):
    """Restoration model for LPNet.

    Paper: Lightweight Pyramid Networks for Image Deraining

    Args:
        generator (dict): Config for the generator structure.
        losses (List[dict]): A list of configs for building losses.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg: (dict or list[dict], optional): Initialization config dict. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 losses,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(generator)

        # loss
        self.loss = nn.ModuleList([
            nn.ModuleDict() for _ in range(self.generator.max_level)
        ])
        for item in losses:
            levels = item.pop('levels')
            for level in levels:
                self.loss[level].update({item['type'].lower(): build_loss(item)})

    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """

        output = self.generator(lq)
        pyr = gaussian_pyramid(gt, sigma=0, n_levels=self.generator.max_level-1,
                               gauss_coeff_backend='cv2')
        pyr.append(gt)

        losses_total = dict()
        for i, losses in enumerate(self.loss):
            for name, loss in losses.items():
                losses_total[f'level{i}-{name}'] = loss(output[i], pyr[i])

        outputs = dict(
            losses=losses_total,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output[-1].cpu()))
        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(lq)[-1]

        if meta is not None and 'pad' in meta[0]:
            output = self.restore_shape(output, meta)
            lq = self.restore_shape(lq, meta)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, ('evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out = self.generator(img)[-1]
        return out
