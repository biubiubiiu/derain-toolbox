import numbers
import os.path as osp

import mmcv

from mmderain.core import crop_border, psnr, ssim, tensor2img

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS


@MODELS.register_module()
class PhysicalModelGuided(BaseModel):
    """Restorer for Physical Model Guided Network.

    Paper: Physical Model Guided Deep Image Deraining

    Args:
        model_cfg (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        loss_weight (tuple[float]): Loss weight for L_{rain}, L_{rainfree}, L_{guide} and L_{p}
        init_cfg: (dict or list[dict], optional): Initialization config dict. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 model_cfg,
                 pixel_loss,
                 loss_weight,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(model_cfg)

        # loss
        self.loss = build_loss(pixel_loss)
        self.loss_weight = loss_weight

    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        R_hat, B_hat, B_refine = self.generator(lq)

        w0, w1, w2, w3 = self.loss_weight

        losses['loss_rain'] = w0 * self.loss(R_hat, lq-gt)
        losses['loss_bg'] = w1 * self.loss(B_hat, gt)
        losses['loss_bg_refine'] = w2 * self.loss(B_refine, gt)
        losses['loss_cycle'] = w3 * self.loss(R_hat+B_hat, lq)

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=B_refine.cpu()))
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
        _, _, output = self.generator(lq)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
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
        outputs = self.generator(img)
        return outputs[-1]
